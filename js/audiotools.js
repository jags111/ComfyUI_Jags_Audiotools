import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "/scripts/api.js";

function audioUpload(node, inputName, inputData, app) {
    const audioWidget = node.widgets.find((w) => w.name === "audio");
    let uploadWidget;

    var default_value = audioWidget.value;
    Object.defineProperty(audioWidget, "value", {
        set : function(value) {
            this._real_value = value;
        },

        get : function() {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_value;
            }

            if (value.filename) {
                let real_value = value;
                value = "";
                if (real_value.subfolder) {
                    value = real_value.subfolder + "/";
                }

                value += real_value.filename;

                if(real_value.type && real_value.type !== "input")
                    value += ` [${real_value.type}]`;
            }
            return value;
        }
    });
    async function uploadFile(file, updateNode, pasted = false) {
        try {
            // Wrap file in formdata so it includes filename
            const body = new FormData();
            body.append("image", file);
            if (pasted) body.append("subfolder", "pasted");
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body,
            });

            if (resp.status === 200) {
                const data = await resp.json();
                // Add the file to the dropdown list and update the widget value
                let path = data.name;
                if (data.subfolder) path = data.subfolder + "/" + path;

                if (!audioWidget.options.values.includes(path)) {
                    audioWidget.options.values.push(path);
                }

                if (updateNode) {
                    audioWidget.value = path;
                }
            } else {
                alert(resp.status + " - " + resp.statusText);
            }
        } catch (error) {
            alert(error);
        }
    }

    const fileInput = document.createElement("input");
    Object.assign(fileInput, {
        type: "file",
        accept: "audio/mp3,audio/wav",
        style: "display: none",
        onchange: async () => {
            if (fileInput.files.length) {
                await uploadFile(fileInput.files[0], true);
            }
        },
    });
    document.body.append(fileInput);

    // Create the button widget for selecting the files
    uploadWidget = node.addWidget("button", "choose file to upload", "Audio", () => {
        fileInput.click();
    });
    uploadWidget.serialize = false;
    return { widget: uploadWidget };
}
ComfyWidgets.AUDIOUPLOAD = audioUpload;

app.registerExtension({
	name: "AudioScheduler.UploadAudio",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name == "LoadAudio") {
			nodeData.input.required.upload = ["AUDIOUPLOAD"];
		}
	},
});

"**************************************************************"
class audiotools {
	constructor() {
		if (!window.__audiotools__) {
			window.__audiotools__ = Symbol("__Jags_Audio__");
		}
		this.symbol = window.__audiotools__;
	}

	getState(node) {
		return node[this.symbol] || {};
	}

	setState(node, state) {
		node[this.symbol] = state;
		app.canvas.setDirty(true);
	}

	addStatusTagHandler(nodeType) {
		if (nodeType[this.symbol]?.statusTagHandler) {
			return;
		}
		if (!nodeType[this.symbol]) {
			nodeType[this.symbol] = {};
		}
		nodeType[this.symbol] = {
			statusTagHandler: true,
		};

		api.addEventListener("Jags_Audio/update_status", ({ detail }) => {
			let { node, progress, text } = detail;
			const n = app.graph.getNodeById(+(node || app.runningNodeId));
			if (!n) return;
			const state = this.getState(n);
			state.status = Object.assign(state.status || {}, { progress: text ? progress : null, text: text || null });
			this.setState(n, state);
		});

		const self = this;
		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx) {
			const r = onDrawForeground?.apply?.(this, arguments);
			const state = self.getState(this);
			if (!state?.status?.text) {
				return r;
			}

			const { fgColor, bgColor, text, progress, progressColor } = { ...state.status };

			ctx.save();
			ctx.font = "12px sans-serif";
			const sz = ctx.measureText(text);
			ctx.fillStyle = bgColor || "dodgerblue";
			ctx.beginPath();
			ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, sz.width + 12, 20, 5);
			ctx.fill();

			if (progress) {
				ctx.fillStyle = progressColor || "green";
				ctx.beginPath();
				ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, (sz.width + 12) * progress, 20, 5);
				ctx.fill();
			}

			ctx.fillStyle = fgColor || "#fff";
			ctx.fillText(text, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
			ctx.restore();
			return r;
		};
	}
}

const audiotools = new audiotools();

app.registerExtension({
	name: "Jags_Audio",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		pysssss.addStatusTagHandler(nodeType);

		if (nodeData.name === "Jags_Audio") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				const r = onExecuted?.apply?.(this, arguments);

				const pos = this.widgets.findIndex((w) => w.name === "tags");
				if (pos !== -1) {
					for (let i = pos; i < this.widgets.length; i++) {
						this.widgets[i].onRemove?.();
					}
					this.widgets.length = pos;
				}

				for (const list of message.tags) {
					const w = ComfyWidgets["STRING"](this, "tags", ["STRING", { multiline: true }], app).widget;
					w.inputEl.readOnly = true;
					w.inputEl.style.opacity = 0.6;
					w.value = list;
				}

				this.onResize?.(this.size);

				return r;
			};
		} else {
			const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
				const r = getExtraMenuOptions?.apply?.(this, arguments);
				let img;
				if (this.imageIndex != null) {
					// An image is selected so select that
					img = this.imgs[this.imageIndex];
				} else if (this.overIndex != null) {
					// No image is selected but one is hovered
					img = this.imgs[this.overIndex];
				}
				if (img) {
					let pos = options.findIndex((o) => o.content === "Save Image");
					if (pos === -1) {
						pos = 0;
					} else {
						pos++;
					}
					options.splice(pos, 0, {
						content: "Jags_Audio",
						callback: async () => {
							let src = img.src;
							src = src.replace("/view?", `/Jags_Audio/tag?node=${this.id}&clientId=${api.clientId}&`);
							const res = await (await fetch(src)).json();
							alert(res);
						},
					});
				}

				return r;
			};
		}
	},
});
