import { app } from "../../scripts/app.js";

const COLOR_THEMES = {
    red: { nodeColor: "#332222", nodeBgColor: "#553333" },
    green: { nodeColor: "#223322", nodeBgColor: "#335533" },
    blue: { nodeColor: "#222233", nodeBgColor: "#333355" },
    pale_blue: { nodeColor: "#2a363b", nodeBgColor: "#3f5159" },
    cyan: { nodeColor: "#223333", nodeBgColor: "#335555" },
    purple: { nodeColor: "#332233", nodeBgColor: "#553355" },
    yellow: { nodeColor: "#443322", nodeBgColor: "#665533" },
    none: { nodeColor: null, nodeBgColor: null } // no color
};

const NODE_COLORS = {
    "JoinAudio": "blue",
    "BatchJoinAudio": "random",
    "CutAudio": "random",
    "DuplicateAudio": "random",
    "StretchAudio": "random",
    "ReverseAudio": "random",
    "ResampleAudio": "random",
    "OTTAudioFX": "random",
    "LoadVST3": "random",
    "BitCrushAudioFX": "random",
    "ChorusAudioFX": "random",
    "ClippingAudioFX": "random",
    "CompressorAudioFX": "random",
    "ConvolutionAudioFX": "random",
    "DelayAudioFX": "random",
    "DistortionAudioFX": "random",
    "GainAudioFX": "random",
    "InvertAudioFX": "random",
    "LimiterAudioFX": "random",
    "MP3CompressorAudioFX": "random",
    "NoiseGateAudioFX": "random",
    "PitchShiftAudioFX": "random",
    "PhaserEffectAudioFX": "random",
    "ReverbAudioFX": "random",
    "HighShelfFilter": "random",
    "HighpassFilter": "random",
    "LadderFilter": "random",
    "LowShelfFilter": "random",
    "LowpassFilter": "random",
    "PeakFilter": "random",
    "GenerateAudioSample": "random",
    "SaveAudioTensor": "blue",
    "LoadAudioFile": "green",
    "PreviewAudioFile": "random",
    "PreviewAudioTensor": "random",
    "GetStringByIndex": "random",
    "LoadAudioModel (DD)": "random",
    "MixAudioTensors": "random",
    "GetAudioFromFolderIndex": "random",
    "ImageToSpectral": "random",
    "PlotSpectrogram": "random",
    "SliceAudio": "random",
    "BatchToList": "random",
    "ConcatAudioList": "random",
    "GenerateAudioWave": "green",
    "SequenceVariation": "random"
 };

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];  // Swap elements
    }
}

let colorKeys = Object.keys(COLOR_THEMES).filter(key => key !== "none");
shuffleArray(colorKeys);  // Shuffle the color themes initially

function setNodeColors(node, theme) {
    if (!theme) {return;}
    node.shape = "box";
    if(theme.nodeColor && theme.nodeBgColor) {
        node.color = theme.nodeColor;
        node.bgcolor = theme.nodeBgColor;
    }
}

const ext = {
    name: "jags.appearance",

    nodeCreated(node) {
        const title = node.getTitle();
        if (NODE_COLORS.hasOwnProperty(title)) {
            let colorKey = NODE_COLORS[title];

            if (colorKey === "random") {
                // Check for a valid color key before popping
                if (colorKeys.length === 0 || !COLOR_THEMES[colorKeys[colorKeys.length - 1]]) {
                    colorKeys = Object.keys(COLOR_THEMES).filter(key => key !== "none");
                    shuffleArray(colorKeys);
                }
                colorKey = colorKeys.pop();
            }

            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

app.registerExtension(ext);