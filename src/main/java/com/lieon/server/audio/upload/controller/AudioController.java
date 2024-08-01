package com.lieon.server.audio.upload.controller;

import com.lieon.server.audio.upload.domain.AudioUploadService;
import com.lieon.server.audio.upload.payload.AudioResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RequestMapping("stt")
@RestController
public class AudioController {
    private AudioUploadService audioUploadService;

    public AudioController(AudioUploadService audioUploadService){
        this.audioUploadService = audioUploadService;
    }

    @PostMapping(value = "/audio", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> handleAudioMessage(@RequestParam("audioFile") MultipartFile audioFile) throws IOException {
        AudioResponse audioResponse;
        try{
            audioResponse = audioUploadService.uploadAudio(audioFile);
        } catch (RuntimeException | IOException e){
            return new ResponseEntity<>(e.getMessage(), HttpStatus.FORBIDDEN);
        }
        return new ResponseEntity<>(audioResponse, HttpStatus.OK);
    }
}
