const db = require('../../models/db')
// Функция для получения favicon, description, title и href для трех последних добавленных сайтов
function getLastThreeSitesInfo() {
    return new Promise((resolve, reject) => {
        const sql = `
            SELECT id, title, description, faviconUrl, url
            FROM sites
            ORDER BY id DESC
            LIMIT 3
        `;
        
        db.all(sql, (err, rows) => {
            if (err) {
                console.error('Ошибка выполнения запроса:', err);
                reject(err);
            } else {
                resolve(rows);
            }
        });
    });
}
module.exports = {getLastThreeSitesInfo}