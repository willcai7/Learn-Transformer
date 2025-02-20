from cs336_basics.src.utils import *
from cs336_basics.src.models import *
import wandb
import time

@dataclass
class TrainingConfig(UtilConfig):
    job_name: Optional[str] = field(default="default")
    dataset_name: Optional[str] = field(default="wikitext-2")

directory = None # directory for outputs and checkpoints
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime()) 
model_name = "GPT2" #model name

# parsing config
parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)
logger = GenLogger(directory, config, raw=raw)

if not raw:
    wandb.init(project=wandb_project, name = timestamp + '-' + model_name, tags=tags, dir=wandb_path)
    wandb.config.update(asdict(config)) 
    checkpoint_path = os.path.join(directory, 'checkpoint.pth')

logging.info(f'Training with config: {asdict(config)}')


# load dataset
dataset = Dataset(**asdict(config))

# load model
model = TransformerLM(**asdict(config))
model = model.to(device)

# loading the optimizer
optimizer = AdamW(model.parameters(), **asdict(config))

# Loss 
loss = nn.CrossEntropyLoss()

# load checkpoint
if init_from != 'scratch':
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]

def eval():
    total_loss = 0
    for _ in range(config.eval_iters):
        x, y = dataset.get_batch('val')
        x, y = x.to(config.device), y.to(config.device)
        with torch.no_grad():
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()
    total_loss /= config.eval_iters
    logging.info(f'Iter: {iter_num}, Val loss: {loss.item():.4f}, LR: {lr:.6f}')
    if config.wandb_logging:
        wandb.log({'val_loss': total_loss, 'lr': lr, 'iter': iter_num})
        save_checkpoint(model, optimizer, iter_num, f'data/out/checkpoints/{config.wandb_run_name}.pt')

curr_time = time.time() 
iter_num = 0
while iter_num < total_iters+1:

    optimizer.zero_grad()

    # backward pass
    x,y = dataset.get_batch('train')
    logits = model(x)
    loss_temp = loss(logits, y)
    loss_temp.backward()

    # update the model
    gradient_clipping(model.parameters(), max_norm)
    lr = get_lr_cosine_schedule(iter_num, lr_max,**asdict(config))
    optimizer.set_lr(lr)
    optimizer.step()
    finish_time = time.time()

    # logging
    if iter_num % log_interval == 0:
        logging.info(f'Iteration: {iter_num}, loss: {loss_temp.item()}, time: {finish_time - curr_time}')
        curr_time = finish_time
    
    # evaluation
    if iter_num % eval_interval == 0:
        eval()
    
    curr_time = finish_time 
    iter_num += 1


logging.shutdown() # close the logger

# upload to s3 bucket
if S3_upload: 
    import s3fs
    s3_file = s3fs.S3FileSystem()
    local_path = directory
    s3_path = S3_bucket_name+f'/GHM/{job_name}/{tree_folder}/{model_name}/{timestamp}'
    s3_file.put(local_path, s3_path, recursive=True) 