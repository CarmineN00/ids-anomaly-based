#include <tflm_esp32.h>
#include <utility.h>
#include "snn_model.h"
#include <eloquent_tinyml.h>

#define ARENA_SIZE 10000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

void setup() {
    Serial.begin(115200);

    while (!tf.begin(snn_model).isOk()) 
        Serial.println(tf.exception.toString());
}


void loop() {
    // classify class 0 (normal)
    if (!tf.predict(x0).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }
    
    Serial.print("expcted class 0, predicted class ");
    Serial.println(tf.classification);
    
    // classify class 1 (DoS)
    if (!tf.predict(x1).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }
    
    Serial.print("expcted class 1, predicted class ");
    Serial.println(tf.classification);
    
    // classify class 2 (Probe)
    if (!tf.predict(x2).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }
    
    Serial.print("expcted class 2, predicted class ");
    Serial.println(tf.classification);

    // how long does it take to run a single prediction?
    Serial.print("It takes ");
    Serial.print(tf.benchmark.microseconds());
    Serial.println("us for a single prediction");
    
    delay(1000);
}