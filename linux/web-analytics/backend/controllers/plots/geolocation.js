const { findEventsByType, findEventByType, findEventByTypeForSite } = require("../../models/event")

const geolocationPlot = async (site_id = undefined) => {
    const events = await (site_id ? findEventByTypeForSite('visitorInfo', site_id) : findEventsByType('visitorInfo'));
    const genderObj =  { "Страна": "Количество пользователей" }

    if (!Array.isArray(events)) {
        console.error("Ожидался массив, получен:", events);
        return []; // Или другая логика обработки ошибки
      }

    events.forEach((el) => {
        const country = JSON.parse(el.eventValue)?.deviceInfo?.ipinfo?.country || "KZ"
        genderObj[country] = genderObj[country] ? genderObj[country] + 1 : 1 
    });

    return Object.entries(genderObj)   
}

module.exports = {
    geolocationPlot
};
