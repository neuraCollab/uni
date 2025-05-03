<?php
require 'vendor/autoload.php';

use League\Csv\Reader;
use League\Csv\Writer;

$csvUrl = 'test.csv';
$downloadAttempts = 10;
$tempDir = __DIR__ . '/temp_files';
$outputFile = __DIR__ . '/final_processed_data.csv';

function downloadFile($url, $file, $attempts): bool {
    for ($i = 1; $i <= $attempts; $i++) {
        echo "Попытка загрузить файл (попытка {$i})...\n";
        try {
            file_put_contents($file, fopen($url, 'r'));
            if (filesize($file) > 0) {
                echo "Файл успешно загружен.\n";
                return true;
            }
        } catch (\Exception $e) {
            echo "Ошибка загрузки: " . $e->getMessage() . "\n";
        }
        sleep(5);
    }
    echo "Не удалось загрузить файл.\n";
    return false;
}

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

function splitCsv(string $inputFile, string $tempDir): array {
    echo "Разделение файла на части...\n";
    try {
        $reader = Reader::createFromPath($inputFile, 'r');
        $reader->setDelimiter(";");
        $reader->setHeaderOffset(0);
        $headers = $reader->getHeader();

        $chunkIndex = 0;
        $chunks = [];
        $chunkRows = [];
        foreach ($reader->getRecords() as $row) {
            $chunkRows[] = $row;
            if (count($chunkRows) >= 10000) {
                $chunkFile = "{$tempDir}/chunk_{$chunkIndex}.csv";
                $chunks[] = $chunkFile;
                saveChunk($chunkRows, $chunkFile);
                $chunkRows = [];
                $chunkIndex++;
            }
        }

        if (!empty($chunkRows)) {
            $chunkFile = "{$tempDir}/chunk_{$chunkIndex}.csv";
            $chunks[] = $chunkFile;
            saveChunk($chunkRows, $chunkFile);
        }

        return $chunks;
    } catch (\Exception $e) {
        echo "Ошибка при разделении CSV: " . $e->getMessage() . "\n";
        return [];
    }
}

function saveChunk(array $rows, string $chunkFile): void {
    echo "Сохранение части в файл {$chunkFile}...\n";
    try {
        $writer = Writer::createFromPath($chunkFile, 'w+');
        $writer->setDelimiter(";");
        foreach ($rows as $row) {
            $writer->insertOne($row);
        }
    } catch (\Exception $e) {
        echo "Ошибка при сохранении части: " . $e->getMessage() . "\n";
    }
}

function processChunk(string $chunkFile, string $tempDir): void {
    echo "Обработка части {$chunkFile}...\n";
    try {
        $reader = Reader::createFromPath($chunkFile, 'r');
        $reader->setDelimiter(";");
        $outputFile = "{$tempDir}/processed_" . basename($chunkFile);
        $output = Writer::createFromPath($outputFile, 'w+');
        $output->setDelimiter(";");

        foreach ($reader->getRecords() as $record) {
            $processedRow = processRow($record);
            $output->insertOne($processedRow);
        }
        echo "Часть {$chunkFile} обработана.\n";
    } catch (\Exception $e) {
        echo "Ошибка при обработке части {$chunkFile}: " . $e->getMessage() . "\n";
    }
}

function combineChunks(array $chunkFiles, string $outputFile, array $headers): void {
    echo "Объединение частей в финальный файл...\n";
    try {
        $output = Writer::createFromPath($outputFile, 'w+');
        $output->setDelimiter(";");
        $output->insertOne($headers);

        foreach ($chunkFiles as $chunkFile) {
            echo "Обработка файла {$chunkFile} для объединения...\n";
            $reader = Reader::createFromPath($chunkFile, 'r');
            $reader->setDelimiter(";");
            $reader->setHeaderOffset(null); // Чтение без заголовков

            $rowCount = 0;
            foreach ($reader->getRecords() as $record) {
                if (!empty(array_filter($record))) { // Проверка на пустую строку
                    $output->insertOne($record);
                    $rowCount++;
                }
            }
            echo "Добавлено строк из {$chunkFile}: {$rowCount}\n";
            if ($rowCount === 0) {
                echo "Предупреждение: файл {$chunkFile} не содержит данных для объединения.\n";
            }
        }
        echo "Все части объединены в {$outputFile}.\n";
    } catch (\Exception $e) {
        echo "Ошибка при объединении частей: " . $e->getMessage() . "\n";
    }
}



function deleteDirectory(string $dir): void {
    if (!is_dir($dir)) {
        return;
    }
    $items = array_diff(scandir($dir), ['.', '..']);
    foreach ($items as $item) {
        $path = "{$dir}/{$item}";
        is_dir($path) ? deleteDirectory($path) : unlink($path);
    }
    rmdir($dir);
}

function mainProcess() {
    global $csvUrl, $downloadAttempts, $tempDir, $outputFile;

    if (!is_dir($tempDir)) {
        mkdir($tempDir, 0777, true);
    }

    $tempCsvFile = "{$tempDir}/data.csv";

    if (downloadFile($csvUrl, $tempCsvFile, $downloadAttempts)) {
        $reader = Reader::createFromPath($tempCsvFile, 'r');
        $reader->setHeaderOffset(0);
        $headers = $reader->getHeader();
        $headersArray = str_getcsv($headers[0], ';');

        $chunks = splitCsv($tempCsvFile, $tempDir);

        $processes = [];
        $outputFiles = [];
        foreach ($chunks as $chunkFile) {
            $outputFile = "{$tempDir}/processed_" . basename($chunkFile);
            $command = "php process_chunk.php " . escapeshellarg($chunkFile) . " " . escapeshellarg($outputFile) . " 2>&1";
            echo "Запуск команды: $command\n";
            $process = popen($command, 'r'); // Открываем процесс для асинхронного выполнения

            if ($process) {
                while (!feof($process)) {
                    echo fgets($process); // Читаем и выводим результат выполнения команды для отладки
                }
                pclose($process);
                if (file_exists($outputFile)) {
                    $outputFiles[] = $outputFile;
                } else {
                    echo "Ошибка: файл {$outputFile} не был создан.\n";
                }
            } else {
                echo "Ошибка при запуске команды: {$command}\n";
            }
        }

        // Проверяем, что все обработанные файлы были созданы, прежде чем объединить их
        if (count($outputFiles) === count($chunks)) {
            combineChunks($outputFiles, $outputFile, $headersArray);
        } else {
            echo "Ошибка: не все чанки были обработаны.\n";
        }

        deleteDirectory($tempDir);
        echo "Временные файлы удалены.\n";
    } else {
        echo "Не удалось загрузить файл после {$downloadAttempts} попыток.\n";
    }
}

mainProcess();



?>
