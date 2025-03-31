const { getVisitorsBySiteId } = require("./visiter");
const db = require("./db")
const {logErrorToFile} = require("../logs/logError")


const createEvent = (visiterId, eventType, eventValue, timestamp) => {
  const sql =
    "INSERT INTO events (visiterId, eventType, eventValue, timestamp) VALUES (?, ?, ?, ?)";
  db.run(sql, [visiterId, eventType, eventValue, timestamp], (err) => {
    if (err) {
      console.error("Ошибка при создании события:", err);
      logErrorToFile(err);
    } else {
      console.log("Событие успешно создано");
    }
  });
};

const findEventByTypeForSite = async (eventType, siteId) => {
  try {
    const visitors = await getVisitorsBySiteId(siteId);
    if (!visitors.length) return []; // Возвращаем пустой массив, если нет посетителей

    const promises = visitors.map(visitor =>
      new Promise((resolve, reject) => {
        db.all('SELECT * FROM events WHERE visiterId = ? AND eventType = ?', [visitor.id, eventType], (err, rows) => {
          if (err) {
            reject(err);
          } else {
            resolve(rows);
          }
        });
      })
    );
    const events = await Promise.all(promises);
    return events.flat(); // Сглаживаем массив результатов
  } catch (err) {
    console.error('Ошибка при выполнении запроса к базе данных:', err);
    logErrorToFile(err);
    throw err;
  }
};


const findEventsByType = (eventType) => {
  return new Promise((resolve, reject) => {
    db.all('SELECT * FROM events WHERE eventType = ?', [eventType], (err, events) => {
      if (err) {
        console.error('Ошибка при выполнении запроса к базе данных:', err);
        logErrorToFile(err);
        reject(err);  // Отклоняем Promise с ошибкой
      } else {
        resolve(events);  // Разрешаем Promise с результатами запроса
      }
    });
  });
};

const findEventsByVisitorId = async (visitorId) => {
  try {
    const events = await db.all('SELECT * FROM events WHERE visiterId = ?', [visitorId]);
    return events;
  } catch (err) {
    console.error('Ошибка при выполнении запроса к базе данных:', err);
    logErrorToFile(err);
    throw err;
  }
};


module.exports = { createEvent, findEventsByType, findEventsByVisitorId, findEventByTypeForSite };
