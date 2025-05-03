<?php
require 'vendor/autoload.php';

use League\Csv\Reader;
use League\Csv\Writer;

function processRow(array $row): array {
    if (!empty($row['images'])) {
        $images = explode(' ', $row['images']);
        $row['images'] = implode(' ', $images);
    }

    if (isset($row['genderSpecific'])) {
        $row['genderSpecific'] = match ($row['genderSpecific']) {
            0 => 'Для двоих',
            1 => 'Для нее',
            2 => 'Для него',
            default => $row['genderSpecific']
        };
    }

    return $row;
}

if ($argc < 3) {
    die("Usage: php process_chunk.php <chunk_file> <output_file>\n");
}

$chunkFile = $argv[1];
$outputFile = $argv[2];

echo "Обработка части {$chunkFile}...\n";
try {
    $reader = Reader::createFromPath($chunkFile, 'r');
    $reader->setDelimiter(";");
    $reader->setHeaderOffset(null); // Читаем данные без заголовков

    $records = $reader->getRecords(); // Читаем данные, пропуская сам заголовок

    // Создаём выходной файл без заголовка
    $output = Writer::createFromPath($outputFile, 'w+');
    $output->setDelimiter(";");

    // Перебираем и обрабатываем каждую строку данных
    foreach ($records as $record) {
        $processedRow = processRow($record);
        $output->insertOne($processedRow);
    }

    echo "Часть {$chunkFile} обработана.\n";
} catch (\Exception $e) {
    echo "Ошибка при обработке части {$chunkFile}: " . $e->getMessage() . "\n";
}
?>
