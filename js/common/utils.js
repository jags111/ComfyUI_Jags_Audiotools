/**
 * File: utils.js
 * Project: comfyui_jags_audiotools
 * Author: jags111
 *
 * Copyright (c) 2023 jags111
 *
 */
import { app } from '../../scripts/app.js'
import { $el } from "../../scripts/ui.js"
import * as shared from './comfy_shared.js'
import { log } from './comfy_shared.js'
import { MetadataDialog } from './modelInfoDialog.js'

// style sheet functions and url functions incl load image
export function addStylesheet(url) {
	if (url.endsWith(".js")) {
		url = url.substr(0, url.length - 2) + "css";
	}
	$el("link", {
		parent: document.head,
		rel: "stylesheet",
		type: "text/css",
		href: url.startsWith("http") ? url : getUrl(url),
	});
}

export function getUrl(path, baseUrl) {
	if (baseUrl) {
		return new URL(path, baseUrl).toString();
	} else {
		return new URL("../" + path, import.meta.url).toString();
	}
}

export async function loadImage(url) {
	return new Promise((res, rej) => {
		const img = new Image();
		img.onload = res;
		img.onerror = rej;
		img.src = url;
	});
}

export function addMenuHandler(nodeType, cb) {

    const GROUPED_MENU_ORDER = {
        "🔄 Swap with...": 0,
        "⛓ Add link...": 1,
        "📜 Add script...": 2,
        "🔍 View model info...": 3,
        "🌱 Seed behavior...": 4,
        "📐 Set Resolution...": 5,
        "✏️ Add 𝚇 input...": 6,
        "✏️ Add 𝚈 input...": 7
    };

    const originalGetOpts = nodeType.prototype.getExtraMenuOptions;

    nodeType.prototype.getExtraMenuOptions = function () {
        let r = originalGetOpts ? originalGetOpts.apply(this, arguments) || [] : [];

        const insertOption = (option) => {
            if (GROUPED_MENU_ORDER.hasOwnProperty(option.content)) {
                // Find the right position for the option
                let targetPos = r.length; // default to the end
                
                for (let i = 0; i < r.length; i++) {
                    if (GROUPED_MENU_ORDER.hasOwnProperty(r[i].content) && 
                        GROUPED_MENU_ORDER[option.content] < GROUPED_MENU_ORDER[r[i].content]) {
                        targetPos = i;
                        break;
                    }
                }
                // Insert the option at the determined position
                r.splice(targetPos, 0, option);
            } else {
                // If the option is not in the GROUPED_MENU_ORDER, simply add it to the end
                r.push(option);
            }
        };

        cb.call(this, insertOption);

        return r;
    };
}

export function findWidgetByName(node, widgetName) {
    return node.widgets.find(widget => widget.name === widgetName);
}

// Utility functions
export function addNode(name, nextTo, options) {
    options = { select: true, shiftX: 0, shiftY: 0, before: false, ...(options || {}) };
    const node = LiteGraph.createNode(name);
    app.graph.add(node);
    node.pos = [
        nextTo.pos[0] + options.shiftX,
        nextTo.pos[1] + options.shiftY,
    ];
    if (options.select) {
        app.canvas.selectNode(node, false);
    }
    return node;
}
// register node and debug functions to test same

function escapeHtml(unsafe) {
    return unsafe
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;')
  }
  app.registerExtension({
    name: 'jags.Debug',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
      if (nodeData.name === 'Debug (mtb)') {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
          const r = onNodeCreated
            ? onNodeCreated.apply(this, arguments)
            : undefined
          this.addInput(`anything_1`, '*')
          return r
        }
  
        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (
          type,
          index,
          connected,
          link_info
        ) {
          const r = onConnectionsChange
            ? onConnectionsChange.apply(this, arguments)
            : undefined
          // TODO: remove all widgets on disconnect once computed
          shared.dynamic_connection(this, index, connected, 'anything_', '*')
  
          //- infer type
          if (link_info) {
            const fromNode = this.graph._nodes.find(
              (otherNode) => otherNode.id == link_info.origin_id
            )
            const type = fromNode.outputs[link_info.origin_slot].type
            this.inputs[index].type = type
            // this.inputs[index].label = type.toLowerCase()
          }
          //- restore dynamic input
          if (!connected) {
            this.inputs[index].type = '*'
            this.inputs[index].label = `anything_${index + 1}`
          }
        }
  
        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
          onExecuted?.apply(this, arguments)
  
          const prefix = 'anything_'
  
          if (this.widgets) {
            // const pos = this.widgets.findIndex((w) => w.name === "anything_1");
            // if (pos !== -1) {
            for (let i = 0; i < this.widgets.length; i++) {
              if (this.widgets[i].name !== 'output_to_console') {
                this.widgets[i].onRemoved?.()
              }
            }
            this.widgets.length = 1
          }
          let widgetI = 1
  
          if (message.text) {
            for (const txt of message.text) {
              const w = this.addCustomWidget(
                MtbWidgets.DEBUG_STRING(`${prefix}_${widgetI}`, escapeHtml(txt))
              )
              w.parent = this
              widgetI++
            }
          }
          if (message.b64_images) {
            for (const img of message.b64_images) {
              const w = this.addCustomWidget(
                MtbWidgets.DEBUG_IMG(`${prefix}_${widgetI}`, img)
              )
              w.parent = this
              widgetI++
            }
            // this.onResize?.(this.size);
            // this.resize?.(this.size)
          }
  
          this.setSize(this.computeSize())
  
          this.onRemoved = function () {
            // When removing this node we need to remove the input from the DOM
            for (let y in this.widgets) {
              if (this.widgets[y].canvas) {
                this.widgets[y].canvas.remove()
              }
              shared.cleanupNode(this)
              this.widgets[y].onRemoved?.()
            }
          }
        }
      }
    },
  })
  
