package com.husercar.djl;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

@RestController
public class ImageDetectController {

    @PostMapping(value = "/upload", produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<String> diagnose(@RequestParam("file") MultipartFile file) throws ModelException, TranslateException, IOException {
        byte[] bytes = file.getBytes();
        Path imageFile = Paths.get(file.getOriginalFilename());
        Files.write(imageFile, bytes);
        return predict(imageFile);
    }


    public ResponseEntity<String> predict(Path imageFile) throws IOException, ModelException, TranslateException {
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone", "resnet50")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                return saveBoundingBoxImage(imageFile, detection);
            }
        }
    }


    private ResponseEntity<String> saveBoundingBoxImage(Path imagePath, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("/usr/src/app");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = createImageWithAlphaChannel(imagePath);
        newImage.drawBoundingBoxes(detection);

        Path imagePath2 = outputDir.resolve("detected.png");
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath2), "png");
        System.out.println("Detected objects image has been saved in" + imagePath2);


        String fileDownloadUri = ServletUriComponentsBuilder.fromCurrentContextPath()
                .path("get")
                .toUriString();
        return ResponseEntity.ok(fileDownloadUri);
    }

    private Image createImageWithAlphaChannel(Path imagePath) throws IOException {
        BufferedImage bufferedImage = ImageIO.read(imagePath.toFile());
    
        // Create a new BufferedImage with alpha channel
        BufferedImage imageWithAlpha = new BufferedImage(bufferedImage.getWidth(),
                bufferedImage.getHeight(), BufferedImage.TYPE_INT_ARGB);
    
        // Draw the JPEG image onto the new image with alpha channel
        Graphics2D g = imageWithAlpha.createGraphics();
        g.drawImage(bufferedImage, 0, 0, null);
        g.dispose();
    
        // Create an ai.djl.modality.cv.Image from the BufferedImage
        return ImageFactory.getInstance().fromImage(imageWithAlpha);
    }


    @GetMapping(value = "/get", produces = MediaType.IMAGE_PNG_VALUE)
    public @ResponseBody byte[] getImageWithMediaType() throws IOException {
        Path imagePath = Paths.get("/usr/src/app/detected.png");  // Update with the correct path
        return Files.readAllBytes(imagePath);
    }



}
