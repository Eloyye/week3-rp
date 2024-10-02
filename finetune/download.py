from huggingface_hub import hf_hub_download

def main():
    hf_hub_download(repo_id="illusin/sql_special", filename="sqllama.Q4_K_M.gguf", local_dir='.')

if __name__ == '__main__':
    main()