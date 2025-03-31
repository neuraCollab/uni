const express = require('express');
const router = express.Router();
const { AddEvent } = require('../controllers/collectController');

// Обработчик POST запросов для создания новой записи
router.post('/', AddEvent);

module.exports = router;
