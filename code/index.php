<?php
// === Detect AJAX-request ===
$is_ajax = isset($_SERVER['HTTP_X_REQUESTED_WITH']) && strtolower($_SERVER['HTTP_X_REQUESTED_WITH']) === 'xmlhttprequest';

function my_error_message() {
    echo "<p style='color:red;'>File or POST-body size overflow.<br><br>Check settings in your php.ini: " . php_ini_loaded_file() . "</p>";
    echo ' - upload_max_filesize: ' . ini_get('upload_max_filesize');
    echo '<br> - post_max_size      : ' . ini_get('post_max_size');
    echo '<br> - memory_limit       : ' . ini_get('memory_limit');
    echo ("<br><br>After modifying some values remember:<br><br> - when you use fpm with php 8.3 'sudo systemctl restart php8.3-fpm'<br><br> - 'else sudo systemctl restart apache2.service'");
}

// === AJAX POST: only return the result ===
if ($_SERVER['REQUEST_METHOD'] === 'POST' && $is_ajax) {
    ob_clean(); // Make sure output buffer is empty and we only send the response
    header('Content-Type: text/html'); // or application/json for JSON

    if (empty($_FILES)) {
        echo "<p>Error receiving file.</p>";
        my_error_message();
        exit;
    }

    if (isset($_FILES['level_image'])) {
        $targetdirectory = __DIR__ . '/uploads/';
        if (!file_exists($targetdirectory)) {
            mkdir($targetdirectory, 0777, true);
        }

        $ext = pathinfo($_FILES['level_image']['name'], PATHINFO_EXTENSION);
        $newName = 'lastfile.' . $ext;
        $targetfile = $targetdirectory . $newName;

        if (move_uploaded_file($_FILES['level_image']['tmp_name'], $targetfile)) {
            $venvpython3 = realpath(__DIR__) . "/my_venv/bin/python3";
            $escapedScript = escapeshellarg(__DIR__ . '/script_test_picture_lite.py');
            $escaped = escapeshellarg($targetfile);
            $cmd = "$venvpython3 $escapedScript $escaped 2>&1 | grep -v INFO";
            $output = shell_exec($cmd);

            echo "<h3>Results:</h3><pre>$output</pre>";
            echo "<img src='uploads/" . htmlspecialchars($newName) . "?" . rand(100000, 999999) . "' height='320' width='80' style='border: 2px dashed black;'>";
        } else {
            echo "<p>Error saving file.</p>";
            my_error_message();
        }
    }

    exit;
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>LevelDetection</title>
    <style>
      #imageInput {
        display: none;
      }

      .custom-file-button {
        display: inline-block;
        padding: 10px;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        cursor: pointer;
        font-size: 40px;
      }

      .custom-file-button:hover {
        background-color: #0056b3;
      }
    </style>
</head>
<body>
<center>
    <h1>Upload an image for level detection</h1>
    <form id="uploadForm" action="" method="post" enctype="multipart/form-data">
        <label id="imageInputLabel" for="imageInput" class="custom-file-button">
          Select or take a picture
        </label>
        <input id="imageInput" type="file" name="level_image" accept="image/*" capture="environment" required>
        <br><br>
        <button type="submit" class="custom-file-button">Send picture</button>
    </form>
    <br>
    <img id="preview" style="max-width: 300px;" />
    <br>
</center>
    <div id="php_page"></div>

    <script>
        // Preview image
        const previewImage = document.getElementById('preview');

        function handleFileSelect(event) {
          const file = event.target.files[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
              previewImage.src = e.target.result;
              previewImage.style.display = 'block';
            };
            reader.readAsDataURL(file);
          }
        }

        document.getElementById('imageInput').addEventListener('change', handleFileSelect);

        // Handle AJAX submit
        const form = document.getElementById('uploadForm');
        form.addEventListener('submit', function(event) {
          event.preventDefault();
          document.getElementById('php_page').innerHTML = '<h2>Sent picture, please wait....</h2>';
          const formData = new FormData(form);

          fetch('', {
            method: 'POST',
            body: formData,
            headers: {
              'X-Requested-With': 'XMLHttpRequest'
            }
          })
          .then(response => response.text())
          .then(data => {
            document.getElementById('php_page').innerHTML = data;
          })
          .catch(error => {
            console.error('Error:', error);
          });
        });

        // New selection clears results
        document.getElementById('imageInputLabel').addEventListener('click', function() {
          document.getElementById('php_page').innerHTML = '';
        });
    </script>
</body>
</html>
