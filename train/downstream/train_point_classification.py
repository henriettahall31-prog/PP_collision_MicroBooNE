#COMPLETELY REDONE FOR MICROBOONE
import os, sys, argparse
sys.path.append('../..')
from fm4npp.utils import YParams
from point_classification_trainer import DownstreamTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--global_log_dir", default='globallogs', type=str)
    args = parser.parse_args()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    trainer = DownstreamTrainer(params, args)
    trainer.launch()

#train from checkpoints
    import os
checkpoint_path = os.path.join('/scratch/midway3/hhall5/checkpoints/pid_microboone', 'pid_training_checkpoint.pth')
if os.path.exists(checkpoint_path):
    print(f"Resuming from {checkpoint_path}")
    trainer.train(pretrain=True, train_from_checkpoint=True, checkpoint_path=checkpoint_path)
else:
    trainer.train(pretrain=True)
