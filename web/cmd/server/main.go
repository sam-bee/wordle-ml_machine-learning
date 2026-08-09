package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/web/internal/server"
)

func main() {
	addr := os.Getenv("WEB_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	inferenceURL := os.Getenv("INFERENCE_URL")
	if inferenceURL == "" {
		inferenceURL = "http://inference:8090"
	}
	handler, err := server.NewHandler(server.Config{InferenceURL: inferenceURL})
	if err != nil {
		log.Fatal(err)
	}

	httpServer := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      125 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	log.Printf("Wordle ML visualiser listening on %s", addr)
	log.Fatal(httpServer.ListenAndServe())
}
