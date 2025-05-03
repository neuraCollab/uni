const db = require("./db")
const {logErrorToFile} = require("../logs/logError")

// Функция для получения посетителя по токену
const getVisitorByToken = (token) => {
  return new Promise((resolve, reject) => {
    db.get("SELECT * FROM visiters WHERE token = ?", [token], (err, row) => {
      if (err) {
        console.error("Ошибка при выполнении запроса к базе данных:", err);
        logErrorToFile(err);
        reject(err); // Отклоняем промис с ошибкой, если есть
      } else {
        resolve(row); // Резолвим промис с результатом
      }
    });
  });
};

// Функция для получения посетителя по токену
const getVisitorById = (visitorId) => {
  return new Promise((resolve, reject) => {
    db.get("SELECT * FROM visiters WHERE id = ?", [visitorId], (err, row) => {
      if (err) {
        console.error("Ошибка при выполнении запроса к базе данных:", err);
        logErrorToFile(err);
        reject(err); // Отклоняем промис с ошибкой, если есть
      } else {
        resolve(row); // Резолвим промис с результатом
      }
    });
  });
};


// Функция для создания посетителя
const createVisitor = (token, siteId) => {
  return new Promise((resolve, reject) => {
    db.run(
      "INSERT INTO visiters (siteId, token) VALUES (?, ?)",
      [siteId, token],
      function (err) {
        if (err) {
          console.error("Ошибка при добавлении посетителя в базу данных:", err);
          logErrorToFile(err);
          reject(err); // Отклоняем промис с ошибкой, если есть
        } else {
          console.log("Посетитель успешно добавлен в базу данных");
          resolve(this.lastID); // Резолвим промис с последним ID
        }
      }
    );
  });
};

// Функция для поиска или создания посетителя по токену
const findOrCreateVisitor = async (token, siteId) => {
  try {
    // Пытаемся найти посетителя по токену
    let visitor = await getVisitorByToken(token);
    if (visitor) {
      // Если посетитель найден, возвращаем его
      return visitor;
    } else {
      // Если посетитель не найден, создаем нового
      const visitorId = await createVisitor(token, siteId);
      // Получаем данные о новом посетителе
      visitor = await getVisitorById(visitorId);
      return visitor;
    }
  } catch (err) {
    console.error("Ошибка при поиске или создании посетителя:", err);
    throw err; // Пробрасываем ошибку дальше
  }
};

const getVisitorsBySiteId = (siteId) => {
  return new Promise((resolve, reject) => {
    db.all("SELECT * FROM visiters WHERE siteId = ?", [siteId], (err, row) => {
      if (err) {
        console.error("Ошибка при выполнении запроса к базе данных:", err);
        logErrorToFile(err);
        reject(err); // Отклоняем промис с ошибкой, если есть
      } else {
        resolve(row); // Резолвим промис с результатом
      }
    });
  });
}


module.exports = { getVisitorByToken, createVisitor, findOrCreateVisitor, getVisitorsBySiteId };
