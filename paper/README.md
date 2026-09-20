# QwerySmith Research Paper

This directory contains the academic paper draft and bibliography for **QwerySmith 1.1**:

> **QwerySmith: Parameter-Efficient Domain Specialization and the Few-Shot Paradox in Small Language Models for Text-to-SQL**  
> *Cyrax et al., Autonomous Intelligence Research, September 2026*

---

## 📁 Directory Structure

```
paper/
├── main.tex                 # Full camera-ready research paper in LaTeX
├── references.bib           # Complete BibTeX bibliography (16 academic citations)
├── figures/                 # Directory holding 13 publication figures (PNG + vector PDF)
│   ├── fig1_execution_accuracy.png
│   ├── fig2_exact_vs_execution.png
│   ├── fig3_clause_f1_scores.png
│   ├── fig4_pairwise_win_loss.png
│   ├── fig5_training_dynamics.png
│   ├── fig6_outcome_transition_matrix.png
│   ├── fig7_clause_confusion_grid.png
│   ├── fig8_inter_system_agreement_matrix.png
│   ├── fig9_complexity_heatmap.png
│   ├── fig10_error_taxonomy_matrix.png
│   ├── fig11_error_migration_matrix.png
│   ├── fig12_clause_correlation_matrices.png
│   └── fig13_token_length_stratification.png
├── tables/                  # Directory holding 10 publication LaTeX tables
│   ├── table1_main_benchmark.tex
│   ├── table2_clause_metrics.tex
│   ├── table3_significance.tex
│   ├── table4_outcome_transition.tex
│   ├── table5_complexity_matrix.tex
│   ├── table6_error_taxonomy.tex
│   ├── table7_diagnostic_clause_matrix.tex
│   ├── table8_error_migration_matrix.tex
│   ├── table9_length_stratification.tex
│   └── table10_persplit_kappa.tex
└── README.md                # This guide
```

---

## 🚀 Compiling the Paper

### Option 1: Overleaf (Recommended)
1. Zip the `paper/` directory:
   ```bash
   zip -r qwerysmith_paper.zip paper/
   ```
2. Go to [Overleaf](https://www.overleaf.com) $\rightarrow$ New Project $\rightarrow$ Upload Project.
3. Select `qwerysmith_paper.zip` and click **Recompile**.

### Option 2: Local Compilation (via pdflatex / tectonic)
If you have TeX Live or MacTeX installed:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using `tectonic`:
```bash
tectonic paper/main.tex
```

---

## 🔄 Regenerating Figures & Tables

To regenerate all 13 figures and 10 tables from your training evaluation runs:
```bash
python paper_eval.py --out /content/drive/MyDrive/qwerysmith-1.1 --paper-dir paper
```
Or in Google Colab with inline rendering:
```python
!python paper_eval.py --out /content/drive/MyDrive/qwerysmith-1.1 --paper-dir paper --display
```
