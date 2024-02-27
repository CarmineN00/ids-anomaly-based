#include <tflm_esp32.h>
#include <utility.h>
#include "snn_model.h"
#include <eloquent_tinyml.h>

#define ARENA_SIZE 10000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

void setup() {

    unsigned long max_time = 0;
    unsigned long min_time = 999999;
    unsigned long total_time = 0;
    int correct = 0;

    Serial.begin(115200);
    
    while (!tf.begin(snn_model).isOk()) 
        Serial.println(tf.exception.toString());

    for (int i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        // Esegui la previsione
        unsigned long start_time = micros();
    
        if (!tf.predict(samples[i]).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }

        // print result
        Serial.print("Expected class ");
        Serial.print(expected_classes[i]);
        Serial.print(", predicted class ");
        Serial.println(tf.classification);

        unsigned long end_time = micros();

        if(expected_classes[i] == tf.classification){ correct++; };
        unsigned long elapsed_time = end_time - start_time;

        if(elapsed_time > max_time){
            max_time = elapsed_time;
          }
        if(elapsed_time < min_time){
            min_time = elapsed_time;
          }
    
        total_time += elapsed_time;
    }

    unsigned long average_time = total_time / sizeof(samples[0]);
    Serial.print("Accuracy: ");
    Serial.print(correct);
    Serial.println("%");

    Serial.print("Massimo tempo di predizione: ");
    Serial.print(max_time);
    Serial.println(" microsecondi");
  
    Serial.print("Minimo tempo di predizione: ");
    Serial.print(min_time);
    Serial.println(" microsecondi");
  
    Serial.print("Tempo medio di predizione: ");
    Serial.print(average_time);
    Serial.println(" microsecondi");
}


void loop() {

}