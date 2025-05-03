const db = require('../../models/db');

function dailyVisitsAndReferrerStats(site_id) {
  return new Promise((resolve, reject) => {
    let analyticsData = {
      dailyVisits: {},
      referrerStats: {},
    };

    // Получаем данные посещений по дням для конкретного сайта
    const visitsByDaySql = `
        SELECT DATE(timestamp) as day, COUNT(*) as count
        FROM events
        JOIN visiters ON events.visiterId = visiters.id
        WHERE visiters.siteId = ?
        GROUP BY day
        ORDER BY day DESC`;

    // Получаем статистику по источникам перехода для конкретного сайта
    const referrerStatsSql = `
        SELECT eventType, COUNT(*) as count
        FROM events
        JOIN visiters ON events.visiterId = visiters.id
        WHERE visiters.siteId = ?
        GROUP BY eventType
        ORDER BY count DESC`;

    db.serialize(() => {
      db.all(visitsByDaySql, [site_id], (err, rows) => {
        if (err) {
          reject(err);
          return;
        }
        analyticsData.dailyVisits = rows.reduce((acc, row) => {
          acc[row.day] = row.count;
          return acc;
        }, {});

        db.all(referrerStatsSql, [site_id], (err, rows) => {
          if (err) {
            reject(err);
            return;
          }
          analyticsData.referrerStats = rows;
          resolve(analyticsData);
        });
      });
    });
  });
}

module.exports = {dailyVisitsAndReferrerStats};
