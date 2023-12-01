import { app } from "/scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from '../../../scripts/widgets.js'
import { $el } from '../../../scripts/ui.js'

let api_host = '127.0.0.1:8188'
let api_base = ''
let url = `http://${api_host}${api_base}`

/* 
A method that returns the required style for the html 
*/
function addPlaybackWidget(node, name, url) {
	let isTick = true;
	const audio = new Audio(url);
	const slider = node.addWidget(
		"slider",
		"loading",
		0,
		(v) => {
			if (!isTick) {
				audio.currentTime = v;
			}
			isTick = false;
		},
		{
			min: 0,
			max: 0,
		}
	);

	const button = node.addWidget("button", `Play ${name}`, "play", () => {
		try {
			if (audio.paused) {
				audio.play();
				button.name = `Pause ${name}`;
			} else {
				audio.pause();
				button.name = `Play ${name}`;
			}
		} catch (error) {
			alert(error);
		}
		app.canvas.setDirty(true);
	});
	audio.addEventListener("timeupdate", () => {
		isTick = true;
		slider.value = audio.currentTime;
		app.canvas.setDirty(true);
	});
	audio.addEventListener("ended", () => {
		button.name = `Play ${name}`;
		app.canvas.setDirty(true);
	});
	audio.addEventListener("loadedmetadata", () => {
		slider.options.max = audio.duration;
		slider.name = `(${audio.duration})`;
		app.canvas.setDirty(true);
	});
}

app.registerExtension({
	name: "jags.PlayAudio",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const AudioPreviews = ["PreviewAudioFile", "PreviewAudioTensor"]
		if (AudioPreviews.includes(nodeData.name)) {
			const WIDGETS = Symbol();
			nodeType.prototype.onExecuted = function (data) {
				if (WIDGETS in this) {
					// Clear all other widgets
					if (this.widgets) {
						this.widgets.length = this[WIDGETS];
					}
					if (this.widgets_values) {
						this.widgets_values.length = this.widgets.length;
					}
				} else {
					// On first execute store widget count
					this[WIDGETS] = this.widgets?.length || 0;
				}

				// For each file create a seek bar + play button
				for (const file of data) {
					addPlaybackWidget(this, file, `/view?type=temp&filename=${encodeURIComponent(file)}`);
				}
			};
		} else if (nodeData.name === "LoadAudioFile") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated?.apply(this, arguments);

				let uploadWidget;
				let pathWidget = this.widgets[0];

				async function uploadFile(file, node) {
					try {
						// Wrap file in formdata so it includes filename
						const body = new FormData();
						body.append("file", file);
						const resp = await fetch("/ComfyUI_Jags_Audiotools/upload/audio", {
							method: "POST",
							body,
						});

					if(node.widgets) {
						node.widgets.length = 1;
					}
					if(node.widgets_values) {
						node.widgets_values.length = 1;
					}
						if (resp.status === 200) {
							const { name } = await resp.json();
							pathWidget.value = name;
							addPlaybackWidget(node, name, `/ComfyUI_Jags_Audiotools/audio?filename=${encodeURIComponent(name)}`)
						} else {
							alert(resp.status + " - " + resp.statusText);
						}
					} catch (error) {
						alert(error);
						throw error;
					}
				}

				const fileInput = document.createElement("input");
				Object.assign(fileInput, {
					type: "file",
					accept: "audio/mpeg,audio/wav,audio/x-wav",
					style: "display: none",
					onchange: async () => {
						if (fileInput.files.length) {
							await uploadFile(fileInput.files[0], this);
						}
					},
				});
				document.body.append(fileInput);

				// Create the button widget for selecting the files
				uploadWidget = this.addWidget("button", "choose file to upload", "audio", () => {
					fileInput.click();
				});
				uploadWidget.serialize = false;

				// Add handler to check if an image is being dragged over our node
				this.onDragOver = function (e) {
					if (e.dataTransfer && e.dataTransfer.items) {
						const file = [...e.dataTransfer.items].find((f) => f.kind === "file" && f.type.startsWith("audio/"));
						return !!file;
					}

					return false;
				};

				// On drop upload files
				this.onDragDrop = function (e) {
					let handled = false;
					for (const file of e.dataTransfer.files) {
						if (file.type.startsWith("audio/")) {
							uploadFile(file, this); 
							handled = true;
						}
					}

					return handled;
				};

				return r;
			};
		}
	},
});
app.registerExtension(node)
