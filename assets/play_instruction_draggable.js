if (!window.dash_clientside) {
  window.dash_clientside = {};
}
window.dash_clientside.clientside = {
  make_draggable: function (is_open, id) {
    if (is_open && id) {
      dragElement(document.getElementById(id));

      function dragElement(elem) {
        let pos1 = 0,
          pos2 = 0,
          pos3 = 0,
          pos4 = 0;
        if (document.getElementById(elem.id + "_header")) {
          document.getElementById(elem.id + "_header").onmousedown =
            dragMouseDown;
        } else {
          elem.onmousedown = dragMouseDown;
        }
        elem.children[0].onmousedown = dragMouseDown;

        function dragMouseDown(e) {
          e.preventDefault();
          pos3 = e.clientX;
          pos4 = e.clientY;
          document.onmouseup = closeDragElement;
          document.onmousemove = elementDrag;
        }

        function elementDrag(e) {
          e.preventDefault();
          pos1 = pos3 - e.clientX;
          pos2 = pos4 - e.clientY;
          pos3 = e.clientX;
          pos4 = e.clientY;
          elem.style.top = elem.offsetTop - pos2 + "px";
          elem.style.left = elem.offsetLeft - pos1 + "px";
        }

        function closeDragElement() {
          document.onmouseup = null;
          document.onmousemove = null;
        }
      }
    }
    return window.dash_clientside.no_update;
  },
};