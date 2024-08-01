package com.lieon.server.audio.upload.domain;

import com.lieon.server.audio.upload.payload.AudioResponse;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.NoSuchElementException;

@RequiredArgsConstructor
@Service
public class AudioUploadService {
    private final Logger logger = LoggerFactory.getLogger(AudioUploadService.class);

    private static final String UPLOAD_DIR = "uploads/audio/";

    public AudioResponse uploadAudio(MultipartFile file) throws IOException {
        if (file.isEmpty()) {
            return new AudioResponse("File is empty!");
        }
        try {
            File uploadFile = new File(UPLOAD_DIR + file.getOriginalFilename());
            file.transferTo(uploadFile);
            return new AudioResponse("File uploaded successfully!");
        } catch (IOException e){
            e.printStackTrace();
            return new AudioResponse("File upload failed!");
        }
    }

}
