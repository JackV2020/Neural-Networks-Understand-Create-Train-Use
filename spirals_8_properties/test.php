<?php

// De map waarin je afbeeldingen staan
$top = './test'; // Pas dit aan naar jouw map

$dirs1 = scandir($top);
$dirs = array_diff($dirs1, array('.', '..'));
echo '<style>';
echo '.image-gallery {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.thumbnail {
    border: 1px solid #ddd;
    padding: 5px;
    max-width: 120px;
    text-align: center;
}
.thumbnail img {
    width: 100%;
    height: auto;
}
';
echo '</style>';
echo '<div class="image-gallery">';
foreach ($dirs as $dir) {
    $dirPath = $top . '/' . $dir;
    if (is_dir($dirPath)) {
        // Haal alle bestanden op uit de map
        $files = scandir($dirPath);

        // Verwijder "." en ".." uit de lijst
        $files = array_diff($files, array('.', '..'));
        foreach ($files as $file) {
            $filePath = $dirPath . '/' . $file;
//            echo $filePath . '<br>-----<br>------<br>-----<br>';
            if (is_file($filePath)) {
                $fileExtension = strtolower(pathinfo($file, PATHINFO_EXTENSION));
                if (in_array($fileExtension, ['jpg', 'jpeg', 'png', 'gif', 'bmp'])) {
                    // Toon afbeelding als thumbnail
                    echo '<div class="thumbnail">';
                    echo '<a href="' . $filePath . '" target="_blank">';
                    echo '<img src="' . $filePath . '" alt="' . $file . '" style="width: 100px; height: 100px; object-fit: cover;" />';
                    echo '</a>';
                    echo '</div>';
                }
            }

        }
    }
}
echo '</div>';

?>
