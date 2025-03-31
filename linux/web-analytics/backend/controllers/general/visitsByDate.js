const db = require('../../models/db')
// Функция для выполнения запроса и возврата промиса с результатами
function getVisitsByDate() {
    return new Promise((resolve, reject) => {
        const sql = `
            SELECT DATE(e.timestamp) AS date,
                   COUNT(v.id) AS visits
            FROM (SELECT id
                  FROM sites
                  LIMIT 3) AS s
            JOIN visiters v ON v.siteId = s.id
            JOIN events e ON e.visiterId = v.id
            GROUP BY DATE(e.timestamp)
            ORDER BY DATE(e.timestamp)
        `;
        
        db.all(sql, (err, rows) => {
            if (err) {
                console.error('Ошибка выполнения запроса:', err);
                reject(err);
            } else {
                const result = [["Дата", "Посещения"]];
                rows.forEach(row => {
                    result.push([row.date, row.visits.toString()]);
                });
                resolve(result);
            }
        });
    });
}


module.exports = {getVisitsByDate};

