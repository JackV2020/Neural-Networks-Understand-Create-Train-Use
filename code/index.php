<!DOCTYPE html>
<html>
<head>
    <title>LevelDetection</title>
</head>
<body>
    <h2>Upload an image for level detection</h2>
    <form action="" method="post" enctype="multipart/form-data">
        <input type="file" name="level_image" accept="image/*" required>
        <br>
        <br>
        <br>
        <button type="submit">Send</button>
    </form>
<?php

function my_error_message() {
    echo "<p style='color:red;'>File or POST-body size overflow.<br><br>Check settings in your php.ini: " . php_ini_loaded_file() . "</p>";
    echo ' - upload_max_filesize: ' . ini_get('upload_max_filesize');
    echo '<br> - post_max_size      : ' . ini_get('post_max_size');
    echo '<br> - memory_limit       : ' . ini_get('memory_limit');
    echo ("<br><br>After modifying some values remember:<br><br> - when you use fpm with php 8.3 'sudo systemctl restart php8.3-fpm' (the path above contains you php version)<br><br> - 'sudo systemctl restart apache2.service'");
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && empty($_FILES)) { 
    echo "<p>Error receiving file.</p>";
    my_error_message() ;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['level_image'])) {
    $targetdirectory = __DIR__ . '/uploads/';
    if (!file_exists($targetdirectory)) {
        mkdir($targetdirectory, 0777, true);
    }
    
    $ext = pathinfo($_FILES['level_image']['name'], PATHINFO_EXTENSION);
    $newName = date('Ymd_His') . '_' . uniqid() . '.' . $ext;
    $newName = 'lastfile.' . $ext;
    $targetfile = $targetdirectory . $newName;
    if (move_uploaded_file($_FILES['level_image']['tmp_name'], $targetfile)) {
        echo "<p>Received image.</p>";
        $venvpython3 = realpath(__DIR__) . "/my_venv/bin/python3";
        $escapedModelLocation = escapeshellarg(__DIR__);
        $escapedScript = escapeshellarg(__DIR__ . '/script_test_image_lite.py');
        $escaped = escapeshellarg($targetfile);
        $cmd = "$venvpython3 $escapedScript $escaped 2>&1 | grep -v INFO";
        $output = $cmd;
        $output = shell_exec($cmd);
        echo "<h3>Results:</h3><pre>$output</pre>";
        echo "<img src='uploads/" . htmlspecialchars($newName) . "' height='320' width='80' style='border: 2px dashed black;'>";

    } else {
        echo "<p>Error saving file.</p>";
        my_error_message();
    }
}
?>
</body>
</html>
