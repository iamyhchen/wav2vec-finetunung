import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer, cer
import argparse

# 用法：python eval_with_hf_model.py --repo_id <username/modelname> --test_path <test.csv> --output_path <output.txt>
def main():
    parser = argparse.ArgumentParser(description="Evaluate Wav2Vec2 Model from Hugging Face Hub")
    parser.add_argument('--checkpoint_path_or_repo_id', type=str, required=True, help='Hugging Face repo id, e.g. username/modelname')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test csv file')
    parser.add_argument('--output_path', type=str, default='pred/predictions.txt', help='Output file path')
    args = parser.parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint_path_or_repo_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint_path_or_repo_id)
    model.eval()

    dataset = load_dataset("csv", data_files={"test": args.test_data_path})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    targets = []
    preds = []

    print("Start evaluation...")
    for sample in dataset["test"]:
        input_values = processor(sample["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        label_str = sample["transcript"]
        # print(f"Label: {label_str}, Prediction: {pred_str}")
        targets.append(label_str)
        preds.append(pred_str)

    # 計算 WER 與 CER
    wer_score = wer(targets, preds)
    cer_score = cer(targets, preds)
    print(f"WER: {wer_score}")
    print(f"CER: {cer_score}")

    # 輸出每一筆結果到 text 檔
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(f"WER: {wer_score}\nCER: {cer_score}\n\n")
        for i in range(len(targets)):
            f.write(f"{i+1}\nLabel: {targets[i]}\nTranscript: {preds[i]}\n\n")
    print(f"evaluate completed and saved to {args.output_path}")

if __name__ == "__main__":
    main()
