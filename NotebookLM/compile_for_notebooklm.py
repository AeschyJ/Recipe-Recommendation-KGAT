import glob
from pathlib import Path

# Get the directory where the script is located (Experiment/NotebookLM)
SCRIPT_DIR = Path(__file__).resolve().parent
# Get the project root directory (Experiment)
PROJECT_ROOT = SCRIPT_DIR.parent


def write_compiled_file(output_filename, title, file_patterns):
    """Helper to find files matching patterns and write them into a single markdown file within the script's directory."""
    compiled_content = f"# {title}\n\n"
    output_path = SCRIPT_DIR / output_filename

    files_processed = 0
    for pattern in file_patterns:
        # Construct the absolute pattern relative to the project root
        search_pattern = str(PROJECT_ROOT / pattern)

        for file_path_str in glob.glob(search_pattern, recursive=True):
            file_path = Path(file_path_str)

            # Skip output files and directories
            if file_path.name.startswith("notebooklm_") or not file_path.is_file():
                continue

            try:
                # Get the path relative to the project root for better display
                relative_path = file_path.relative_to(PROJECT_ROOT)

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                compiled_content += f"## File: {relative_path}\n\n"

                ext = file_path.suffix
                if ext == ".py":
                    compiled_content += f"```python\n{content}\n```\n\n"
                elif ext in [".txt", ".json", ".csv", ".log", ".bat"]:
                    compiled_content += f"```text\n{content}\n```\n\n"
                else:
                    compiled_content += f"{content}\n\n"

                files_processed += 1
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    if files_processed > 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(compiled_content)
        print(f"Successfully compiled {files_processed} files to {output_path.name}")
    else:
        print(f"No files found for {output_path.name}, skipping.")


def compile_docs():
    # 1. Code: train (包含 run 腳本)
    write_compiled_file(
        output_filename="notebooklm_code_run.md",
        title="Source Code: Training & Run Scripts",
        file_patterns=[
            "main.py",
            "simplify_output_data.py",
            "run_experiments.bat",
            "src/train_att.py",
            "src/train_bi_interaction.py",
            "src/run_explainer_*.py",
            "src/generate_explanations.py",
        ],
    )

    # 2. Code: model (包含 explainer)
    write_compiled_file(
        output_filename="notebooklm_code_model.md",
        title="Source Code: Explainer & Common Models",
        file_patterns=["src/model/explainer.py", "src/model/explainer_attention.py"],
    )

    # 3. Code: KGAT
    write_compiled_file(
        output_filename="notebooklm_code_kgat.md",
        title="Source Code: KGAT Attention Model",
        file_patterns=["src/model/kgat_attention.py"],
    )

    # 4. Code: Bi-Interaction
    write_compiled_file(
        output_filename="notebooklm_code_bi_interaction.md",
        title="Source Code: Bi-Interaction Model",
        file_patterns=["src/model/kgat_bi_interaction.py"],
    )

    # 5. Log
    write_compiled_file(
        output_filename="notebooklm_log.md",
        title="Experiment Logs (Simplified)",
        file_patterns=["output/simplified_for_llm/logs/**/*.txt"],
    )

    # 6. Explanation
    write_compiled_file(
        output_filename="notebooklm_explanation.md",
        title="Experiment Explanations (Simplified)",
        file_patterns=[
            # "output/simplified_for_llm/explanations/**/*.txt",
            # "output/simplified_for_llm/explanations/**/*.json",
            "output/fidelity/**/*.txt",
            "output/fidelity/**/*.json",
        ],
    )

    # 7. Docs: ADR
    write_compiled_file(
        output_filename="notebooklm_docs_adr.md",
        title="Architecture Decision Records (ADR)",
        file_patterns=["docs/adr/**/*.md"],
    )

    # 8. Docs: Other (Architecture, API, etc.)
    write_compiled_file(
        output_filename="notebooklm_docs_architecture.md",
        title="System Architecture & Other Docs",
        file_patterns=["docs/*.md"],
    )


if __name__ == "__main__":
    # Clean up old monolithic file if it exists in script dir
    old_monolithic = SCRIPT_DIR / "notebooklm_reference_data.md"
    if old_monolithic.exists():
        old_monolithic.unlink()
        print(f"Removed old {old_monolithic.name}")

    compile_docs()
