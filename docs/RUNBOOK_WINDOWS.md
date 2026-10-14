# Windows runbook (PowerShell)

A step-by-step path from a clean Windows 10/11 machine to a finished pair comparison. Commands
are PowerShell. Every file path below is an example: change it to match your machine.

Everything the harness does is local: it talks to `llama-server` on `127.0.0.1`. Nothing is sent
to the internet by this repository (downloads of weights and of the behaviours file are steps you
run yourself).

## 0. What you need

| Item | Notes |
|---|---|
| Python 3.10 to 3.13 | `py -0p` lists the installed versions |
| `llama.cpp` for Windows | Prebuilt release zips contain `llama-server.exe`, `llama-quantize.exe`; the conversion script `convert_hf_to_gguf.py` comes from the source tree |
| A GPU build is optional | Without a GPU, drop `-ngl 99` and expect runs to take hours |
| Disk | About 8 GB per 4B pair (f16 + Q4_K_M for both arms) |

## 1. Install the harness

```powershell
cd C:\work\refusalscope
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass     # only for this window
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q                                                      # mock-only, no GPU needed
refusalscope mock-demo --out $env:TEMP\rs-demo                 # MOCK-labelled end-to-end run
```

If `Activate.ps1` is blocked, the `Set-ExecutionPolicy` line above is the fix (it changes policy
for the current PowerShell process only).

## 2. Build the pair from one checkpoint

Use the same pipeline for both arms. Quantisation differences between arms confound the study.

```powershell
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-4B --local-dir C:\models\hf\qwen3-4b-stock
# Run Heretic on the stock checkpoint (follow https://github.com/p-e-w/heretic for the current
# install and command) and save the ablated weights to C:\models\hf\qwen3-4b-ablated.
# Write down the Heretic version and the parameters it reports; put them in the config notes.

$Convert = "C:\tools\llama.cpp\convert_hf_to_gguf.py"
$Quant   = "C:\tools\llama.cpp\llama-quantize.exe"
foreach ($arm in "stock", "ablated") {
    python $Convert "C:\models\hf\qwen3-4b-$arm" --outfile "C:\models\qwen3-4b-$arm-f16.gguf" --outtype f16
    & $Quant "C:\models\qwen3-4b-$arm-f16.gguf" "C:\models\qwen3-4b-$arm-Q4_K_M.gguf" Q4_K_M
}
```

## 3. Serve both arms

```powershell
$Llama = "C:\tools\llama.cpp\llama-server.exe"

Start-Process $Llama -ArgumentList @(
  "-m", "C:\models\qwen3-4b-stock-Q4_K_M.gguf", "--alias", "qwen3-4b-stock",
  "--host", "127.0.0.1", "--port", "8080", "--jinja", "--ctx-size", "8192", "--parallel", "1", "-ngl", "99")

Start-Process $Llama -ArgumentList @(
  "-m", "C:\models\qwen3-4b-ablated-Q4_K_M.gguf", "--alias", "qwen3-4b-ablated",
  "--host", "127.0.0.1", "--port", "8081", "--jinja", "--ctx-size", "8192", "--parallel", "1", "-ngl", "99")
```

Check that both answer: `Invoke-RestMethod http://127.0.0.1:8080/v1/models` and the same for
port 8081. Wait until each window prints that it is listening.

Why these flags: `--jinja` makes the server parse tool calls with the model's chat template
(without it tool calls come back as text and the harness reports a serving problem);
`--ctx-size 8192` fits the largest requests (16 tool definitions, multi-step agent transcripts);
`--parallel 1` stops the server from splitting that window across slots.

If the two servers do not fit in memory together, serve one pair at a time and pass
`--model <name>` (twice) to `refusalscope run`; run the next pair into the same `--run-id`.

## 4. Get the behaviours file for the ablation check

The repository bundles no harmful content. Download a public file yourself and keep it out of git
(`data\` is ignored):

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest -OutFile data\harmful_behaviors.csv `
  -Uri https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
```

## 5. Configure and check

```powershell
Copy-Item configs\example_qwen3.json configs\my_study.json
notepad configs\my_study.json         # set quant, params_b, ports, notes (Heretic version)
refusalscope check-config --config configs\my_study.json
refusalscope doctor       --config configs\my_study.json
```

`doctor` must show no `FAIL` lines. Each failure prints the fix; the same cases are in
[the troubleshooting flowchart](FLOWCHARTS.md).

## 6. Run, analyse, report

```powershell
refusalscope run --config configs\my_study.json --suite all --run-id qwen3-pilot-1
```

Expect minutes to hours depending on hardware: Suite A is 374 requests per model, Suite B up to
about 2,300 per model (378 cases x 2 conditions x a few steps), plus 100 refusal probes, 60 KL
probes and 100 capability items. Close the window or press Ctrl-C at any time: the same command
with the same `--run-id` resumes and retries only the jobs that failed.

```powershell
refusalscope analyze --run runs\qwen3-pilot-1 --out results
refusalscope report  --results results --out report --update-readme README.md
Get-ChildItem report
```

Open `report\qwen3-pilot-1-report.md` and the SVG figures. Read the "Ablation check" table first:
if a pair is `ablation-unverified`, nothing else about it is a confirmatory result.

## 7. Windows-specific problems

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `refusalscope` not recognised | The venv is not active; run `.\.venv\Scripts\Activate.ps1`, or call `python -m refusalscope` |
| `connection refused` in `doctor` | The server window closed or is still loading; check the port; Windows Defender Firewall may prompt the first time `llama-server.exe` listens, allow private networks |
| `doctor` says the tool call was written as plain text | The server was started without `--jinja` |
| `exceeds the available context size` | Raise `--ctx-size` (16384) and keep `--parallel 1` |
| Garbled unicode in the console | `chcp 65001` or `$OutputEncoding = [Text.UTF8Encoding]::new()`; files on disk are UTF-8 either way |
| Long paths fail | Keep the repository and models near the drive root, or enable long paths in Windows |
| Output redirected with `>` looks odd in Windows PowerShell 5.1 | It writes UTF-16; use `Out-File -Encoding utf8` or PowerShell 7 |
