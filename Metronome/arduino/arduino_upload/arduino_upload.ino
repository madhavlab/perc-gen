/*
 * Created by ArduinoGetStarted.com
 *
 * This example code is in the public domain
 *
 * Tutorial page: https://arduinogetstarted.com/tutorials/arduino-force-sensor
 */

#define FORCE_SENSOR_PIN A2 // the FSR and 10K pulldown are connected to A2

void setup() {
  Serial.begin(9600);
}

void loop() {
  int analogReading = analogRead(FORCE_SENSOR_PIN);

//  Serial.print("Force sensor reading = ");
  Serial.print(analogReading); // print the raw analog reading
  Serial.print("\n");


//  if (analogReading < 10)       // from 0 to 9
//    Serial.println(" -> no pressure");
//  else if (analogReading < 200) // from 10 to 199
//    Serial.println(" -> light touch");
//  else if (analogReading < 500) // from 200 to 499
//    Serial.println(" -> light squeeze");
//  else if (analogReading < 800) // from 500 to 799
//    Serial.println(" -> medium squeeze");
//  else // from 800 to 1023
//    Serial.println(" -> big squeeze");

  delay(5);
}
