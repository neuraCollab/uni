function generateUniqueId() {
  const timestamp = new Date().getTime().toString(16); // Время в шестнадцатеричном формате
  const randomStr = Math.random().toString(16).slice(2); // Случайное число в шестнадцатеричном формате
  return timestamp + randomStr;
}

var uniqueToken = generateUniqueId()

// Функция для отправки данных на сервер
function sendData(data) {
  // Используйте AJAX запрос для отправки данных
  // Например, с помощью Fetch API или XMLHttpRequest
  // Замените URL на ваш адрес сервера для обработки данных
  fetch("http://localhost:3001/collect", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ...data, token: uniqueToken }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json(); // Если сервер возвращает JSON
    })
    .then((data) => {
      console.log("Success:", data); // Обработка данных, полученных от сервера
    })
    .catch((error) => console.error("Error sending data:", error));
}

// Функция для отслеживания геопозиции пользователя
function trackGeoLocation() {
  // Проверяем поддержку геолокации в браузере
  if ("geolocation" in navigator) {
    navigator.geolocation.getCurrentPosition((position) => {
      // Отправляем данные о геопозиции на сервер
      sendData({
        type: "geolocation",
        type_value: `${position.coords.latitude} - ${position.coords.longitude}`,
        timestamp: new Date().toISOString(),
      });
    });
  } else {
    console.log("Geolocation is not supported by this browser");
  }
}

// Функция для отслеживания источника перехода на сайт
function trackReferrer() {
  // Получаем URL реферера (источника перехода на сайт)
  const referrer = document.referrer;

  // Отправляем данные о реферере на сервер
  sendData({
    type: "referrer",
    type_value: referrer,
    timestamp: new Date().toISOString(),
  });
}

// Функция для отслеживания кликов по странице
function trackClicks() {
  // Добавляем обработчик события клика для всего документа
  document.addEventListener("click", function (event) {
    // Отправляем данные о клике на сервер
    sendData({
      type: "click",
      type_value: event.target.tagName,
      timestamp: new Date().toISOString(),
    });
  });
}

// Функция для отслеживания времени, проведенного на сайте
function trackTimeOnSite() {
  // Устанавливаем время начала посещения сайта
  const startTime = new Date().getTime();

  // Добавляем обработчик события unload для отслеживания ухода пользователя со страницы
  window.addEventListener("beforeunload", function () {
    // Вычисляем время, проведенное на сайте
    const endTime = new Date().getTime();
    const timeOnSite = endTime - startTime;

    // Отправляем данные о времени на сервер
    sendData({
      type: "timeOnSite",
      type_value: timeOnSite,
      timestamp: new Date().toISOString(),
    });
  });
}

// Отслеживание загрузки файлов
function trackFileDownloads() {
  var links = document.querySelectorAll("a");
  links.forEach(function (link) {
    if (link.getAttribute("href").match(/\.(zip|pdf|doc|xls)$/i)) {
      link.addEventListener("click", function (event) {
        // Отправка данных на сервер о загрузке файла
        sendData({
          type: "fileDownload",
          type_value: link.getAttribute("href"),
          timestamp: new Date().toISOString(),
        });
      });
    }
  });
}

// Функция для отслеживания отправки всех форм на странице
function trackAllFormSubmits() {
  var forms = document.querySelectorAll("form");
  forms.forEach(function (form) {
    form.addEventListener("submit", function (event) {
      // Получаем имя формы, если оно задано
      var formName = form.getAttribute("name") || "Unnamed Form";
      // Отправка данных на сервер о отправке формы
      sendData({
        type: "formSubmit",
        type_value: formName,
        timestamp: new Date().toISOString(),
      });
    });
  });
}

// Функция для отслеживания переходов по ссылкам
function trackLinkClicks() {
  var links = document.querySelectorAll("a");
  links.forEach(function (link) {
    link.addEventListener("click", function (event) {
      // Отправка данных на сервер о клике по ссылке
      sendData({
        type: "linkClick",
        type_value: link.href,
        timestamp: new Date().toISOString(),
      });
    });
  });
}

function getBrowserName(userAgent) {
  // Порядок важен, также возможно ложное срабатывание для браузеров не включённых в список

  if (userAgent.includes("Firefox")) {
    // "Mozilla/5.0 (X11; Linux i686; rv:104.0) Gecko/20100101 Firefox/104.0"
    return "Mozilla Firefox";
  } else if (userAgent.includes("SamsungBrowser")) {
    // "Mozilla/5.0 (Linux; Android 9; SAMSUNG SM-G955F Build/PPR1.180610.011) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/9.4 Chrome/67.0.3396.87 Mobile Safari/537.36"
    return "Samsung Internet";
  } else if (userAgent.includes("Opera") || userAgent.includes("OPR")) {
    // "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36 OPR/90.0.4480.54"
    return "Opera";
  } else if (userAgent.includes("Edge")) {
    // "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299"
    return "Microsoft Edge (Legacy)";
  } else if (userAgent.includes("Edg")) {
    // "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36 Edg/104.0.1293.70"
    return "Microsoft Edge (Chromium)";
  } else if (userAgent.includes("Chrome")) {
    // "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
    return "Google Chrome or Chromium";
  } else if (userAgent.includes("Safari")) {
    // "Mozilla/5.0 (iPhone; CPU iPhone OS 15_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Mobile/15E148 Safari/604.1"
    return "Apple Safari";
  } else {
    return "unknown";
  }
}

function getDeviceType() {
  const userAgent = navigator.userAgent.toLowerCase();
  const isMobile =
    /mobile|iphone|ipad|ipod|android|blackberry|mini|windows\sce|palm/i.test(
      userAgent
    );

  if (isMobile) {
    return "mobile";
  } else {
    return "desktop";
  }
}

async function getVisitorInfo() {
  // Получаем User-Agent строки браузера
  var userAgent = navigator.userAgent.toLowerCase();

  // Определяем пол по ключевым словам в User-Agent строке
  var gender = "Не определено";
  if (
    userAgent.includes("male") ||
    userAgent.includes("man") ||
    userAgent.includes("boy")
  ) {
    gender = "male";
  } else if (
    userAgent.includes("female") ||
    userAgent.includes("woman") ||
    userAgent.includes("girl")
  ) {
    gender = "female";
  }

  // Определяем возраст по ключевым словам в User-Agent строке
  var age = "Не определено";
  if (userAgent.includes("age")) {
    var ageRegex = /\bage\s+(\d+)/;
    var matches = userAgent.match(ageRegex);
    if (matches && matches.length > 1) {
      age = matches[1];
    }
  }

  const response = await fetch("https://ipinfo.io/json");

  const data = await response.json();

  // Получаем технические данные устройства
  var deviceInfo = {
    browser: getBrowserName(navigator.userAgent),
    deviceType: getDeviceType(),
    ipinfo: JSON.stringify(data) || "Не определено",
    location:
      data.city + ", " + data.region + ", " + data.country || "Не определено",
    screenWidth: window.screen.width,
    screenHeight: window.screen.height,
    colorDepth: window.screen.colorDepth,
  };

  // Отправляем данные на сервер
  sendData({
    type: "visitorInfo",
    type_value: {
      gender: gender,
      age: age,
      deviceInfo: deviceInfo,
    },

    timestamp: new Date().toISOString(),
  });
}

// Вызываем функции отслеживания событий при загрузке страницы
document.addEventListener("DOMContentLoaded", function () {
  trackGeoLocation();
  trackFileDownloads();
  trackReferrer();
  trackAllFormSubmits();
  trackLinkClicks();
  getVisitorInfo();
  trackClicks();
  trackTimeOnSite();
});
