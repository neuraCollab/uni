const db = require("../../models/db");

function getTimeOnSiteForGender(gender) {
  return new Promise((resolve, reject) => {
    const sql = `
            SELECT eventValue
            FROM events e
            JOIN visiters v ON e.visiterId = v.id
            WHERE e.eventType = 'visitorInfo'
        `;

    db.all(sql, (err, rows) => {
      if (err) {
        console.error("Ошибка выполнения запроса:", err);
        reject(err);
      } else {
        let totalDuration = 0;
        let count = 0;
        rows.forEach((row) => {
          if (typeof row.eventValue === "object") {
            const eventValue = JSON.parse(row.eventValue);
            if (eventValue.gender === gender && eventValue.timeOnSite) {
              totalDuration += eventValue.timeOnSite;
              count++;
            }
          }
        });

        const avgTimeOnSite = count > 0 ? totalDuration / count : 0;
        resolve({
          gender: gender,
          avgTimeOnSite: avgTimeOnSite,
        });
      }
    });
  });
}


function getTotalTimeOnSite() {
    return new Promise((resolve, reject) => {
      const sql = `
        SELECT SUM(CAST(eventValue AS INTEGER)) AS total_time
        FROM events
        WHERE eventType = 'timeOnSite';
      `;
  
      db.get(sql, (err, row) => {
        if (err) {
          console.error('Ошибка при выполнении запроса:', err);
          reject(err);
        } else {
          resolve( [["Гендер", "Время"], ["Не определено", String(row.total_time / 1000)]]);
        }
      });
    });
  }
async function getTimeOnSiteForFemaleVisitors() {
  return await getTimeOnSiteForGender("female");
}

async function getTimeOnSiteForMaleVisitors() {
  return await getTimeOnSiteForGender("male");
}

async function getTimeOnSiteForUndefinedVisitors() {
  return await getTimeOnSiteForGender("Не определено");
}

async function getGenderData() {
  const data = [
    await getTimeOnSiteForFemaleVisitors(),
    await getTimeOnSiteForMaleVisitors(),
    await getTimeOnSiteForUndefinedVisitors(),
  ];
  return data.map(el => {
    const key = el['gender']
    return [key, el['avgTimeOnSite']]
  })
}

module.exports = {
    getTotalTimeOnSite
};
