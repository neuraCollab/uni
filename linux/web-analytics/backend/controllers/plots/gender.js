const { findEventByTypeForSite, findEventsByType }  = require("../../models/event");

const genderTo3dPiechart = async (site_id = undefined) => {
    const events = await (site_id ? findEventByTypeForSite('visitorInfo', site_id) : findEventsByType('visitorInfo'));
    const genderObj = { "Гендр": "Количество людей" };
  
    if (!Array.isArray(events)) {
      console.error("Ожидался массив, получен:", events);
      return []; // Или другая логика обработки ошибки
    }
  
    events.forEach((el) => {
      const eventValue = JSON.parse(el.eventValue);
      const gender = eventValue.gender;
      genderObj[gender] = genderObj[gender] ? genderObj[gender] + 1 : 1;
    });
  
    return Object.entries(genderObj);
  };
  

module.exports = {
    genderTo3dPiechart
};
