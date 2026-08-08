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

	httpServer := &http.Server{
		Addr:              addr,
		Handler:           server.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("Wordle ML splash page listening on %s", addr)
	log.Fatal(httpServer.ListenAndServe())
}
