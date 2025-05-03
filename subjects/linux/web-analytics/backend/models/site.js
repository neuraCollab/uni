const db = require("./db")
const {logErrorToFile} = require("../logs/logError")
// Функция для создания сайта
const createSite = (url, title, description, faviconUrl) => {
  return new Promise((resolve, reject) => {
    const sql = "INSERT INTO sites (url, title, description, faviconUrl) VALUES (?, ?, ?, ?)";
    db.run(sql, [url, title, description, faviconUrl], (err) => {
      if (err) {
        console.error("Ошибка при создании сайта:", err);
        logErrorToFile(err);
        resolve(false); // Ошибка, возвращаем false
      } else {
        console.log("Сайт успешно создан");
        resolve(true); // Успех, возвращаем true
      }
    });
  });
};

// Функция для поиска сайта по его URL
const findSiteByUrl = (url) => {
  return new Promise((resolve, reject) => {
    const sql = "SELECT * FROM sites WHERE url = ?";
    db.get(sql, [url], (err, row) => {
      if (err) {
        console.error("Ошибка при поиске сайта:", err);
        logErrorToFile(err);
        reject(err);  // Отклоняем Promise с ошибкой
      } else {
        resolve(row); // Разрешаем Promise с результатом запроса
      }
    });
  });
};

const findSiteById = (id) => {
  return new Promise((resolve, reject) => {
    const sql = "SELECT * FROM sites WHERE id = ?";
    db.get(sql, [id], (err, row) => {
      if (err) {
        console.error("Ошибка при поиске сайта:", err);
        logErrorToFile(err);
        reject(err); // Отклоняем промис с ошибкой
      } else {
        resolve(row); // Разрешаем промис с найденным объектом
      }
    });
  });
};

module.exports = { findSiteByUrl, createSite, findSiteById };
