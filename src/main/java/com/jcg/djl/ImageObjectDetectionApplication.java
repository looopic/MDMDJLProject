package com.kaviddiss.djldemo;


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
import org.apache.commons.compress.utils.IOUtils;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@SpringBootApplication
@RestController
public class ImageObjectDetectionApplication {

    public static void main(String[] args) {
        SpringApplication.run(ImageObjectDetectionApplication.class, args);
    }

    @PostMapping(value = "/upload", produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<String> diagnose(@RequestParam("file") MultipartFile file) throws ModelException, TranslateException, IOException {
        ImageObjectDetectionApplication imageObjectDetectionApplication = new ImageObjectDetectionApplication();
        byte[] bytes = file.getBytes();
        Path imageFile = Paths.get(file.getOriginalFilename());
        Files.write(imageFile, bytes);
        return imageObjectDetectionApplication.predict(imageFile);
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
                return saveBoundingBoxImage(img, detection);
            }
        }
    }


    private ResponseEntity<String> saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("src/main/resources");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected.png");
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath), "png");
        System.out.println("Detected objects image has been saved in" + imagePath);


        String fileDownloadUri = ServletUriComponentsBuilder.fromCurrentContextPath()
                .path("get")
                .toUriString();
        return ResponseEntity.ok(fileDownloadUri);

    }

    @GetMapping(
            value = "/get",
            produces = MediaType.IMAGE_JPEG_VALUE
    )
    public @ResponseBody byte[] getImageWithMediaType() throws IOException {
        InputStream in = new ClassPathResource(
                "detected.png").getInputStream();
        return IOUtils.toByteArray(in);
    }

}
