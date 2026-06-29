# amalo

## Estudo forte de sharpness

O script `strong_sharpness_study.py` roda uma comparacao mais justa entre SGD,
Adam, SAM-SGD e SAM-Adam para estudar minimos agudos/planos via
`lambda_max` da Hessiana.

Smoke test rapido:

```bash
python strong_sharpness_study.py --quick --dataset cifar10 --device cpu
```

Experimento recomendado para CIFAR-10:

```bash
python strong_sharpness_study.py --dataset cifar10 --epochs 100 --seeds 0 1 2 3 4
```

Extensao para CIFAR-100:

```bash
python strong_sharpness_study.py --dataset cifar100 --epochs 150 --n-train 50000 --n-test 10000 --seeds 0 1 2 3 4
```

Saidas principais em `figuras/strong_sharpness/<dataset>_<model>/`:

- `runs.csv`: metricas finais por otimizador e seed.
- `summary.csv`: media e desvio padrao.
- `matched_runs.csv` e `matched_summary.csv`: analise filtrada para comparar
  apenas solucoes com acuracia de treino parecida dentro de cada seed.
- `lambda_history.csv`: evolucao de `lambda_max`.
- `dominant_eigen_cut.csv`: corte `L(theta + alpha v_max)`.
- `lambda_por_epoca.png`, `lambda_vs_gap_loss.png`,
  `corte_autovetor_dominante.png`: graficos para o relatorio.
