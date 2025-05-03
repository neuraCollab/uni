const { findEventByTypeForSite, findEventsByType }  = require("../../models/event");

const deviceColumnDiagram = async (site_id = undefined) => {
    const events = await (site_id ? findEventByTypeForSite('visitorInfo', site_id) : findEventsByType('visitorInfo'));
    const genderObj =  { "Девайс": "Количество посещений" }

    if (!Array.isArray(events)) {
        console.error("Ожидался массив, получен:", events);
        return []; // Или другая логика обработки ошибки
      }
      

    events.forEach((el) => {
        const device = JSON.parse(el.eventValue)?.deviceInfo?.deviceType || "desktop"
        genderObj[device] = genderObj[device] ? genderObj[device] + 1 : 1 
    });

    return Object.entries(genderObj)

}

module.exports = {
    deviceColumnDiagram
};
