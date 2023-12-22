/**
 * File: playAudio.js
 * Project: comfyui_jags_audiotools
 * Author: jags111
 *
 * Copyright (c) 2023 jags111
 *
 */

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

// SoundPlayer.js

function SoundPlayer() {
    this.addInput("audio_file_path", "string"); // Input for file path
    this.addInput("volume", "number");          // Input for volume
    this.addOutput("action", "string");         // Output to trigger actions

    this.properties = { 
        audio_file_path: "", 
        volume: 1.0 
    };

    // Define default size
    this.size = [200, 100];
    // Adding a message handler
    this.addMessageHandler("custom_event", this.onCustomEvent);
}


SoundPlayer.title = "Jags_SoundPlayer";
SoundPlayer.desc = "Controls audio playback";


// Executed every frame
SoundPlayer.prototype.onExecute = function() {
    const path = this.getInputData(0) || this.properties.audio_file_path;
    const volume = this.getInputData(1) !== null ? this.getInputData(1) : this.properties.volume;

    // Reset action
    this.setOutputData(0, null);
};



// Assuming `app` is your application instance
app.registerExtension({
    name: "Jags_audio.AudioHelpers.Jags_SoundPlayer",
    nodeCreated(node) {
        const title = node.getTitle();
        switch (title) {
            case "PLAY":
                // Custom styling for PLAY nodes
                break;
            case "STOP":
                // Custom styling for STOP nodes
                break;
            case "PAUSE":
                // Custom styling for PAUSE nodes
                break;
        }
    }
});



// Function to trigger specific actions
SoundPlayer.prototype.triggerAction = function(action) {
    this.setOutputData(0, action);
};

// Adding widgets to the node
SoundPlayer.prototype.onDrawBackground = function(ctx) {
    // Drawing background elements if needed
};
// Custom message event handler
SoundPlayer.prototype.onCustomEvent = function(msg) {
    // Handle the custom event
    console.log("Received custom event:", msg);
    // You can add logic here to do something based on the message
};

// Event handler for node being added to the graph
SoundPlayer.prototype.onAdded = function() {
    console.log("SoundPlayer node added to the graph.");
    // Additional actions when the node is added to the graph
};

// Event handler for node being removed from the graph
SoundPlayer.prototype.onRemoved = function() {
    console.log("SoundPlayer node removed from the graph.");
    // Cleanup or additional actions when the node is removed
};

// Register in LiteGraph
//LiteGraph.registerNodeType("audio/soundPlayer", SoundPlayer);

// Register the node in LiteGraph
LiteGraph.registerNodeType("Jags_Audio", SoundPlayer);

// message handler
function myMessageHandler(event) {
    alert(event.detail.something);
}
// in setup()
window.addEventListener("my-message-handle", myMessageHandler);



SoundPlayer.prototype.onDrawForeground = function(ctx) {
    if (this.widgets) return;

    this.addWidget("button", "Play", null, () => { this.triggerAction("play"); });
    this.addWidget("button", "Pause", null, () => { this.triggerAction("pause"); });
    this.addWidget("button", "Stop", null, () => { this.triggerAction("stop"); });
    this.addWidget("button", "Load", null, () => { this.triggerAction("load"); });
    this.addWidget("button", "Save", null, () => { this.triggerAction("save"); });
};
