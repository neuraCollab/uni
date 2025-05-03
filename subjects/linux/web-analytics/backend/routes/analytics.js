const express = require('express');
const router = express.Router();
const analyticsController = require('../controllers/analyticsController');

// Обработчик POST запросов для создания новой записи
router.post('/', analyticsController.createRecord);

// Обработчик GET запросов для получения всех записей
router.get('/', analyticsController.getAllRecords);

// Обработчик GET запросов для получения записи по id
router.get('/:id', analyticsController.getRecordById);

// Обработчик PUT запросов для обновления записи по id
router.put('/:id', analyticsController.updateRecordById);

// Обработчик DELETE запросов для удаления записи по id
router.delete('/:id', analyticsController.deleteRecordById);

module.exports = router;
