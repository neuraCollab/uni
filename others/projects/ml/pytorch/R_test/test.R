# quick_plot.R — мгновенный график в RStudio

# Используем встроенный датасет
data(cars)

# Строим график — он сразу появится во вкладке "Plots"
plot(
  cars$speed, cars$dist,
  main = "Тормозной путь vs Скорость",
  xlab = "Скорость (миль/ч)",
  ylab = "Тормозной путь (футы)",
  col = "darkgreen",
  pch = 19,
  cex = 1.2
)

# Добавим линию регрессии
abline(lm(dist ~ speed, data = cars), col = "red", lwd = 2)

# Готово! График уже отображается.