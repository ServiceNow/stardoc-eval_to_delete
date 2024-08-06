import torch
import torch.distributed as dist
import torch.multiprocessing as tmp


from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer
import random
from transformers import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    # Args that only apply to Video Dataset
    parser.add_argument('--nframe', type=int, default=8)
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--use-subtitle', action='store_true')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='.', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Using a [the first (or) random] sample_size out of each benchmark if needed
    parser.add_argument('--sample_size', type=int, default=-1, help='Number of samples of each benchmark to use. Default is -1, using all of the samples from each benchmark.')
    parser.add_argument('--random', default=False, action='store_true', help='If set to "true", will use a random subset of size "--sample_size" out of each benchmark, else will use the first "--sample_size" samples instead.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducable random selection of samples out of benchmarks')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Rerun: will remove all evaluation temp files
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()

    return args

def seed_everything(seed):

    # Set seed for torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for random
    random.seed(seed)

    # Set seed for transformers
    set_seed(seed)

def sample_dataset(dataset, sample_size, is_random):
    dataframe = dataset.data
    dataset_name = dataset.dataset_name
    print(f'Sampling {sample_size} samples from {dataset_name}')
    if dataset_name.lower() == 'mme':
        # choose the (sample_size / 14) first rows out of each category
        if sample_size % 14 != 0:
            raise Exception('MME has 14 categories, so the sample size should be divisible by 14.')
        sub_sample_size = sample_size / 14
        to_return = dataframe.groupby('category').head(sub_sample_size)
    else:
        if is_random:
            indices = np.random.choice(np.arange(0, len(dataframe)), size=sample_size, replace=False)
            to_return = dataframe.iloc[indices]
        else:
            to_return = dataframe.iloc[:sample_size]
    
    assert len(to_return) == sample_size
    return to_return

def run_on_gpu(rank, world_size, benchmark_chunk, kwargs):
    
    # kwargs.pop('benchmark_list')
    # kwargs.pop('benchmark_chunks')
    # kwargs.pop('benchmark_chunk')
    print(f"Process {rank} started.")
    try:
        # Initialize the process group for distributed training
        torch.distributed.init_process_group(
            backend='nccl',                  # Use NCCL backend for NVIDIA GPUs
            init_method='tcp://127.0.0.1:12355', # Initialization method and address
            world_size=world_size,           # Total number of processes (GPUs)
            rank=rank                        # Rank of the current process
        )
        print(f"Process {rank} initialized process group.")

        # Set the CUDA device to the GPU corresponding to the current process rank
        torch.cuda.set_device(rank)
        print(f"Process {rank} set CUDA device.")

        # Process the data chunk on the assigned GPU
        result = run_model_on_benchmark(process_rank=rank, benchmark_list=benchmark_chunk[rank], **kwargs)
        print(f"Process {rank} completed data processing.")

        # Clean up the process group
        torch.distributed.destroy_process_group()
        print(f"Process {rank} destroyed process group.")

        return result
    
    except Exception as e:
        print(f"Process {rank} encountered an error: {e}")


def run_model_on_benchmark(process_rank, args, model, model_name, sample_size, benchmark_list, logger):
    # run_model_on_benchmark(args, model, benchmark_list, logger)

     
    # args = kwargs["args"]
    # model = kwargs["model"]
    # model_name = kwargs["model_name"]
    # sample_size = kwargs["sample_size"]
    # benchmark_chunks = kwargs["benchmark_list"]
    # logger = kwargs["logger"]
    

    pred_root = osp.join(args.work_dir, model_name)

    os.makedirs(pred_root, exist_ok=True)
    print(benchmark_list)
    for _, dataset_name in enumerate(benchmark_list):
            if dataset_name.lower != 'mme':
                is_random = args.random
            else:
                is_random = False

            print(f'{process_rank} Processing {dataset_name}')
            dataset_kwargs = {}
            if dataset_name == 'MMLongBench_DOC':
                dataset_kwargs['model'] = args.model_name
            if dataset_name == 'MMBench-Video':
                dataset_kwargs['pack'] = args.pack
            if dataset_name == 'Video-MME':
                dataset_kwargs['use_subtitle'] = args.use_subtitle

            
            dataset = build_dataset(dataset_name, **dataset_kwargs)

            if dataset is None:
                logger.error(f'{process_rank} Dataset {dataset_name} is not valid, will be skipped. ')
                continue
            
            if args.sample_size != -1:
                dataset.data = sample_dataset(dataset, args.sample_size, is_random)

            result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'
            if dataset_name in ['MMBench-Video']:
                packstr = 'pack' if args.pack else 'nopack'
                result_file = f'{pred_root}/{model_name}_{dataset_name}_{args.nframe}frame_{packstr}.xlsx'
            if dataset_name in ['Video-MME']:
                if args.pack:
                    logger.info(f'{process_rank} Video-MME not support Pack Mode, directly change to unpack')
                    args.pack = False
                packstr = 'pack' if args.pack else 'nopack'
                subtitlestr = 'subs' if args.use_subtitle else 'nosubs'
                result_file = f'{pred_root}/{model_name}_{dataset_name}_{args.nframe}frame_{packstr}_{subtitlestr}.xlsx'

            if dataset.TYPE == 'MT':
                result_file = result_file.replace('.xlsx', '.tsv')

            if sample_size != -1:
                if is_random:
                    result_file = result_file.replace(f'{dataset_name}', f'{dataset_name}_sampleSize{sample_size}_randomSeed{args.seed}')
                else:
                    result_file = result_file.replace(f'{dataset_name}', f'{dataset_name}_sampleSize{sample_size}')

            if osp.exists(result_file) and args.rerun:
                for keyword in ['openai', 'gpt', 'auxmatch']:
                    if sample_size != -1:
                        if is_random:
                            os.system(f'rm {pred_root}/{model_name}_{dataset_name}_sampleSize{sample_size}_randomSeed{args.seed}_{keyword}*')
                        else:
                            os.system(f'rm {pred_root}/{model_name}_{dataset_name}_sampleSize{sample_size}_{keyword}*')
                    else:    
                        os.system(f'rm {pred_root}/{model_name}_{dataset_name}_{keyword}*')

            if model is None:
                model = model_name  # which is only a name

            # Perform the Inference
            if dataset.MODALITY == 'VIDEO':
                model = infer_data_job_video(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    nframe=args.nframe,
                    pack=args.pack,
                    verbose=args.verbose,
                    subtitle=args.use_subtitle,
                    api_nproc=args.nproc)
            elif dataset.TYPE == 'MT':
                model = infer_data_job_mt(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=args.nproc,
                    ignore_failed=args.ignore,
                    sample_size=sample_size,
                    random_seed=args.seed if is_random else None)
            else:
                model = infer_data_job(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=args.nproc,
                    ignore_failed=args.ignore,
                    sample_size=sample_size,
                    random_seed=args.seed if is_random else None)

            # Set the judge kwargs first before evaluation or dumping
            judge_kwargs = {
                'nproc': args.nproc,
                'verbose': args.verbose,
            }
            if args.retry is not None:
                judge_kwargs['retry'] = args.retry
            if args.judge is not None:
                judge_kwargs['model'] = args.judge
            else:
                if dataset.TYPE in ['MCQ', 'Y/N']:
                    judge_kwargs['model'] = 'chatgpt-0125'
                elif listinstr(['MMVet', 'MathVista', 'LLaVABench', 'MMBench-Video', 'MathVision'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4-turbo'
                elif listinstr(['MMLongBench', 'MMDU'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'
            if 'OPENAI_API_KEY_JUDGE' in os.environ and len(os.environ['OPENAI_API_KEY_JUDGE']):
                judge_kwargs['key'] = os.environ['OPENAI_API_KEY_JUDGE']
            if 'OPENAI_API_BASE_JUDGE' in os.environ and len(os.environ['OPENAI_API_BASE_JUDGE']):
                judge_kwargs['api_base'] = os.environ['OPENAI_API_BASE_JUDGE']

            
            if dataset_name in ['MMMU_TEST']:
                result_json = MMMU_result_transfer(result_file)
                logger.info(f'{process_rank} Transfer MMMU_TEST result to json for official evaluation, '
                            f'json file saved in {result_json}')  # noqa: E501
                continue
            elif 'MMT-Bench_ALL' in dataset_name:
                submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                logger.info(f'{process_rank} Extract options from prediction of MMT-Bench FULL split for official evaluation '
                            f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                            f'submission file saved in {submission_file}')  # noqa: E501
                continue
            elif 'MLLMGuard_DS' in dataset_name:
                logger.info(f'{process_rank} The evaluation of MLLMGuard_DS is not supported yet. ')  # noqa: E501
                continue
            elif 'AesBench_TEST' == dataset_name:
                logger.info(f'{process_rank} The results are saved in {result_file}. '
                            f'Please send it to the AesBench Team via huangyipo@hotmail.com.')  # noqa: E501
                continue

            if dataset_name in [
                'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
            ]:
                if not MMBenchOfficialServer(dataset_name):
                    logger.error(
                        f'Can not evaluate {dataset_name} on non-official servers, '
                        'will skip the evaluation. '
                    )
                    continue

            eval_proxy = os.environ.get('EVAL_PROXY', None)
            old_proxy = os.environ.get('HTTP_PROXY', '')

            if eval_proxy is not None:
                proxy_set(eval_proxy)

            eval_results = dataset.evaluate(result_file, **judge_kwargs)
            if eval_results is not None:
                assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                logger.info(f'{process_rank} The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                logger.info(f'{process_rank} Evaluation Results:')
            if isinstance(eval_results, dict):
                logger.info('\n' + json.dumps(eval_results, indent=4))
            elif isinstance(eval_results, pd.DataFrame):
                if len(eval_results) < len(eval_results.columns):
                    eval_results = eval_results.T
                logger.info('\n' + tabulate(eval_results))

            if eval_proxy is not None:
                proxy_set(old_proxy)

 
def main():
    logger = get_logger('RUN')

    args = parse_args()
    seed_everything(args.seed)
    assert len(args.data), '--data should be a list of data files'

    if args.retry is not None:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))
    
    sample_size = args.sample_size
    # todo problematic benchmarks with subsampling: ['MME', 'VCR_EN_EASY_100']
    
    for _, model_name in enumerate(args.model):
        model = None

        world_size = torch.cuda.device_count()  # Number of available GPUs

        print('\n\n***** SETTING WORLD_SIZE TO 1 MANUALLY ******* \n\n')
        world_size = 1

        if world_size == 0:
            raise ValueError("No GPUs available. Check your CUDA installation and GPU setup.")

        # Split data into chunks, one for each GPU
        benchmark_chunks = [args.data[i::world_size] for i in range(world_size)]
        print(f"Data chunks: {benchmark_chunks}")

        kwargs = {
            "args": args,
            "model": model,
            "model_name": model_name,
            "sample_size": sample_size,
            "logger": logger
        }

        start = datetime.datetime.now()

        # Spawn processes, each running the `run_on_gpu` function
        tmp.spawn(
            run_on_gpu,           # Target function to run in parallel
            args=(world_size, benchmark_chunks, kwargs),  # Arguments for the target function
            nprocs=world_size,    # Number of processes to spawn (one per GPU)
            join=True             # Wait for all processes to complete
        )

        end = datetime.datetime.now()

        print(f'Process took {(end - start).total_seconds()} seconds ({(end - start).total_seconds() // 3600} hours and {((end - start).total_seconds() % 3600) // 60} minutes) with {world_size} number of GPUS!')
        logger.info(f'Process took {(end - start).total_seconds()} seconds ({(end - start).total_seconds() // 3600} hours and {((end - start).total_seconds() % 3600) // 60} minutes) with {world_size} number of GPUS!')



        
if __name__ == '__main__':
    load_env()
    main()
    
