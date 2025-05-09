package controllers

import (
	"fmt"
	"net/http"

	mqtt "github.com/eclipse/paho.mqtt.golang"
)

var mqttClient mqtt.Client

// Initialize the MQTT Client
func InitMqttClient() {
	opts := mqtt.NewClientOptions()
	opts.AddBroker("tcp://broker.emqx.io:1883") // Use the EMQX broker
	opts.SetClientID("go-backend-client")

	mqttClient = mqtt.NewClient(opts)
	if token := mqttClient.Connect(); token.Wait() && token.Error() != nil {
		panic(token.Error())
	}
	fmt.Println("Connected to MQTT broker")
}

// Publish message to MQTT to control LED
func publishMessage(topic, message string) {
	token := mqttClient.Publish(topic, 0, false, message)
	token.Wait()
	fmt.Printf("Published message: %s to topic: %s\n", message, topic)
}

// Handler to turn LED on
func TurnLedOn(w http.ResponseWriter, r *http.Request) {
	publishMessage("embedded_backend", "1") // Publish "1" to turn LED ON
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "LED turned ON")
}

// Handler to turn LED off
func TurnLedOff(w http.ResponseWriter, r *http.Request) {
	publishMessage("embedded_backend", "0") // Publish "0" to turn LED OFF
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "LED turned OFF")
}
