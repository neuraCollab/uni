const { findEventByTypeForSite, findEventsByType }  = require("../../models/event");

const getRefererData = async (site_id = undefined) => {
    const events = await (site_id ? findEventByTypeForSite('referrer', site_id) : findEventsByType('referrer'));
    const referrerObj = { "Откуда пришли": "Количество людей" };
  
    if (!Array.isArray(events)) {
      console.error("Ожидался массив, получен:", events);
      return []; // Или другая логика обработки ошибки
    }
  
    events.forEach((el) => {
      const referrer = el.eventValue;
      referrerObj[referrer] = referrerObj[referrer] ? referrerObj[referrer] + 1 : 1;
    });
  
    return Object.entries(referrerObj);
  };
  
  module.exports = {getRefererData};