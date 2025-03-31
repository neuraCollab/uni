const express = require('express');
const router = express.Router();
const  mainController = require('../controllers/mainController');

// Обработчик GET запросов для получения всех записей
router.get('/', mainController.getAllData);


module.exports = router;
