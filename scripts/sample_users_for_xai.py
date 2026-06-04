import argparse
import json
import os
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="隨機抽取使用者供 XAI 解釋與 Fidelity 測試")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory containing stats.pkl")
    parser.add_argument("--num_users", type=int, default=500, help="Number of users to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output/sampled_users.json", help="Output JSON file path")
    return parser.parse_args()

def main():
    args = parse_args()

    stats_path = os.path.join(args.data_dir, "stats.pkl")
    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} 不存在。")
        return

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    n_users = stats["n_users"]

    # 設定隨機種子以確保可重複性
    np.random.seed(args.seed)
    print(f"隨機種子: {args.seed}")

    # 隨機抽取不重複的 user_id
    if args.num_users > n_users:
        print(f"警告：要求的數量 ({args.num_users}) 大於全部使用者數量 ({n_users})。將抽取全部使用者。")
        sampled_users = np.arange(n_users).tolist()
    else:
        sampled_users = np.random.choice(n_users, args.num_users, replace=False).tolist()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled_users, f, indent=2)

    print(f"成功隨機抽取了 {len(sampled_users)} 位使用者，並儲存至 {args.output}")
    print(f"Sampled Users: {sampled_users}")

if __name__ == "__main__":
    main()
