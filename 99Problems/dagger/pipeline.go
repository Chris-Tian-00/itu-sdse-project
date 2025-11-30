package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	if err := RunPipeline(ctx); err != nil {
		log.Fatalf("Pipeline failed: %v", err)
	}
}

// RunPipeline runs all Python stages in order
func RunPipeline(ctx context.Context) error {
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer()))
	if err != nil {
		return err
	}
	defer client.Close()

	// use Python 3.10 slim container
	container := client.Container().
		From("python:3.10-slim").
		WithMountedDirectory("/app", client.Host().Directory("99Problems/Module1")).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "--upgrade", "pip"}).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}) // we need to make one

	// run all scripts in order (from src folder)
	scripts := []string{
		"src/01_load.py",
		"src/02_feature_selection.py",
		"src/03_clean_separate.py",
		"src/04_combine_bin_save.py",
		"src/05_train.py",
		"src/06_model_selection.py",
		"src/07_deploy.py",
	}


	for _, script := range scripts {
		log.Println("Running", script)
		container = container.WithExec([]string{"python", script})
	}

	// export the artifacts folder with final model(s)
	_, err = container.Directory("artifacts").Export(ctx, "model")
	if err != nil {
		return err
	}

	log.Println("Pipeline completed")
	return nil
}
