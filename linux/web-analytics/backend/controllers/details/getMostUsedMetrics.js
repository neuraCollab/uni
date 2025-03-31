const db = require('../../models/db')

function getMostUsedMetrics(siteId) {
  return new Promise((resolve, reject) => {
    const sql = `
        SELECT eventValue
        FROM events
        JOIN visiters ON events.visiterId = visiters.id
        WHERE visiters.siteId = ?
      `;

    db.all(sql, [siteId], (err, rows) => {
      if (err) {
        reject(err);
      } else {
        const counts = {
          screenWidth: {},
          screenHeight: {},
          colorDepth: {},
        };

        // Обрабатываем каждую строку и обновляем подсчеты
        rows.forEach((row) => {
          try {
            const deviceInfo = (row.eventValue[0] === "{" && row.eventValue) ? JSON.parse(row.eventValue)?.deviceInfo : {}
            console.log(row.eventValue);

            ["screenWidth", "screenHeight", "colorDepth"].forEach((metric) => {
              const value = deviceInfo[metric];
              if (value) {
                counts[metric][value] = (counts[metric][value] || 0) + 1;
              }
            });
          } catch (error) {
            console.error("Ошибка при разборе данных:", error);
          }
        });

        // Выбираем наиболее часто встречающееся значение для каждой метрики
        const metrics = {};
        Object.keys(counts).forEach((metric) => {
          const sortedValues = Object.keys(counts[metric]).sort(
            (a, b) => counts[metric][b] - counts[metric][a]
          );
          metrics[metric] = sortedValues[0] || null;
        });

        resolve(metrics);
      }
    });
  });
}

module.exports = {getMostUsedMetrics};
