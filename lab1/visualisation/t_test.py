import math


def calculate_mean_and_variance(data: list[float]) -> tuple[float, float, int]:
    """Вычисляет среднее значение, несмещенную дисперсию и размер выборки."""
    n = len(data)
    if n < 2:
        raise ValueError("Размер выборки должен быть не менее 2")

    mean = sum(data) / n
    # Несмещенная дисперсия (Bessel's correction: делим на n - 1)
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)

    return mean, variance, n


def ndtr(x: float) -> float:
    """Встроенная аппроксимация функции распределения стандартного нормального закона (CDF)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def perform_pure_python_ttest(
        agent_a_name: str,
        agent_b_name: str,
        data_a: list[float],
        data_b: list[float],
        metric_name: str,
        alpha: float = 0.05
) -> dict:
    """
    Выполняет двухвыборочный t-критерий Уэлча без использования scipy.
    Использует Z-аппроксимацию p-value, что абсолютно точно для N = 300.
    """
    # 1. Считаем базовые статистики
    mean_a, var_a, n_a = calculate_mean_and_variance(data_a)
    mean_b, var_b, n_b = calculate_mean_and_variance(data_b)

    # 2. Вычисляем t-статистику по формуле Уэлча
    # t = (M1 - M2) / sqrt(V1/N1 + V2/N2)
    denominator = math.sqrt((var_a / n_a) + (var_b / n_b))

    if denominator == 0:
        t_stat = 0.0
    else:
        t_stat = (mean_a - mean_b) / denominator

    # 3. Вычисляем p-value (двусторонний тест)
    # Так как N_A = 300 и N_B = 300, число степеней свободы > 500.
    # При таких объемах распределение Стьюдента неотличимо от нормального распределения.
    p_value = 2.0 * (1.0 - ndtr(abs(t_stat)))

    significant = p_value < alpha

    # 4. Формируем текстовый отчет
    std_a = math.sqrt(var_a)
    std_b = math.sqrt(var_b)

    report = (
        f"=== Т-критерий для метрики: {metric_name} ===\n"
        f"Уровень значимости alpha = {alpha}\n\n"
        f"Группа A ({agent_a_name}): Ср. = {mean_a:.4f}, СКО = {std_a:.4f}, N = {n_a}\n"
        f"Группа B ({agent_b_name}): Ср. = {mean_b:.4f}, СКО = {std_b:.4f}, N = {n_b}\n\n"
        f"t-статистика = {t_stat:.4f}\n"
        f"p-value = {p_value:.6f}\n"
        f"Статистически значимые различия: {'ДА' if significant else 'НЕТ'}\n"
    )

    if significant:
        better_agent = agent_a_name if mean_a < mean_b else agent_b_name
        report += f"Вывод: Агент {better_agent} показал статистически значимо лучший результат.\n"
    else:
        report += "Вывод: Различия статистически не значимы, принимается нулевая гипотеза.\n"

    print(report)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_significant": int(significant),
        "text_report": report
    }
