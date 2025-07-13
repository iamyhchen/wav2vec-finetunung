import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingface_hub import login

# 使用 argparse 取得模型路徑、repo_id、token
parser = argparse.ArgumentParser(description="Push trained Wav2Vec2 model to Hugging Face Hub")
parser.add_argument('--model_dir', type=str, required=True, help='Path to trained model directory')
parser.add_argument('--repo_id', type=str, required=True, help='Hugging Face repo id, e.g. username/modelname')
parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face access token')
args = parser.parse_args()

# 登入 Hugging Face Hub
login(token=args.hf_token)

# 推送模型
print(f"Pushing model from {args.model_dir} to {args.repo_id} ...")
model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
model.push_to_hub(args.repo_id)
processor.push_to_hub(args.repo_id)
print("Model and processor pushed to HuggingFace Hub!")
