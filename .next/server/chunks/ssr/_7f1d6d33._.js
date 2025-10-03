module.exports = [
"[project]/lib/layout.shared.tsx [app-rsc] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "baseOptions",
    ()=>baseOptions
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$rsc$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/server/route-modules/app-page/vendored/rsc/react-jsx-dev-runtime.js [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$image$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/image.js [app-rsc] (ecmascript)");
;
;
function baseOptions() {
    return {
        nav: {
            title: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$rsc$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$rsc$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["Fragment"], {
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$rsc$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$image$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["default"], {
                        className: "rounded-full",
                        src: "/logo.svg",
                        width: 20,
                        height: 20,
                        alt: "logo"
                    }, void 0, false, {
                        fileName: "[project]/lib/layout.shared.tsx",
                        lineNumber: 16,
                        columnNumber: 13
                    }, this),
                    "ML Specialization"
                ]
            }, void 0, true)
        },
        // see https://fumadocs.dev/docs/ui/navigation/links
        links: []
    };
}
}),
"[project]/lib/source.ts [app-rsc] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "getLLMText",
    ()=>getLLMText,
    "getPageImage",
    ()=>getPageImage,
    "source",
    ()=>source
]);
var __TURBOPACK__imported__module__$5b$project$5d2f2e$source$2f$index$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/.source/index.ts [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$fumadocs$2d$core$2f$dist$2f$source$2f$index$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/fumadocs-core/dist/source/index.js [app-rsc] (ecmascript) <locals>");
;
;
const source = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$fumadocs$2d$core$2f$dist$2f$source$2f$index$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__$3c$locals$3e$__["loader"])({
    baseUrl: '/ml',
    source: __TURBOPACK__imported__module__$5b$project$5d2f2e$source$2f$index$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["docs"].toFumadocsSource()
});
function getPageImage(page) {
    const segments = [
        ...page.slugs,
        'image.png'
    ];
    return {
        segments,
        url: `/og/ml/${segments.join('/')}`
    };
}
async function getLLMText(page) {
    const processed = await page.data.getText('processed');
    return `# ${page.data.title} (${page.url})

${processed}`;
}
}),
"[project]/public/images/M2/multiple-regression-notations.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/multiple-regression-notations.fe70bc6f.png");}),
"[project]/public/images/M2/multiple-regression-notations.png.mjs { IMAGE => \"[project]/public/images/M2/multiple-regression-notations.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$multiple$2d$regression$2d$notations$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/multiple-regression-notations.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$multiple$2d$regression$2d$notations$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1005,
    height: 503,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAbklEQVR42h2Myw6DIADA/P/v20GjM+DIwhaBgOOlLJ3z0FObduohGPsb/TAw3WfGacJaRy2ZWjOdty9W1bNqiX2rC2ctxym/rdLFYHB6ZvOG/PGUGEgp0trOsZ9BzpFlEQgpeWqNMYby35eNkgI/2sp4y9qmOfoAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/model-with-n-features.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/model-with-n-features.576c16fb.png");}),
"[project]/public/images/M2/model-with-n-features.png.mjs { IMAGE => \"[project]/public/images/M2/model-with-n-features.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$model$2d$with$2d$n$2d$features$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/model-with-n-features.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$model$2d$with$2d$n$2d$features$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1005,
    height: 441,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAYUlEQVR42h2MSQ4DIRAD+f8/J5doCARB0wuq9OTgg+Vylet6cdeKqqFmmDke8c/TyzPU/uXTO2MMlqwcnUgg4lCaGG07opoWQfZEZDITnBoU89Sdk7RxdiPWG183Kp2e5x8LTnx+GAjSjgAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M2/multiple-linear-regression.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/multiple-linear-regression.adbb529a.png");}),
"[project]/public/images/M2/multiple-linear-regression.png.mjs { IMAGE => \"[project]/public/images/M2/multiple-linear-regression.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$multiple$2d$linear$2d$regression$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/multiple-linear-regression.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$multiple$2d$linear$2d$regression$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 999,
    height: 492,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAXUlEQVR42k3MWw6DMBBD0ex/py1VVZJmHpmQy8AXlo7kD8vFY1K78BOjXtRpPrG5uFLmsRAzmip/dyIcjcdAR9DzQSWZZK+I9KR4DMqxFiNHpoN9b7xfn9u2fXEfnDdYfIuJkdB7AAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M2/vectorization.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/vectorization.4e38a6bb.png");}),
"[project]/public/images/M2/vectorization.png.mjs { IMAGE => \"[project]/public/images/M2/vectorization.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$vectorization$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/vectorization.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$vectorization$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1003,
    height: 502,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAcklEQVR42g3NwRKCIAAAUf7/46pLKSrDBEhSiIDj2EyHjcMe9vSEsY5na4mJepxsueL8wtb+WwpCjwN9d6cfJGb2ePdCdxqvPXnNiGlUDFJirWEtFeMcSilCeLPvB+LRSa6XG9OkKCkRPhHrZmKjyvnjD2WrdeiMH4T0AAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M2/vectorization-bts.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/vectorization-bts.6020ab2b.png");}),
"[project]/public/images/M2/vectorization-bts.png.mjs { IMAGE => \"[project]/public/images/M2/vectorization-bts.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$vectorization$2d$bts$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/vectorization-bts.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$vectorization$2d$bts$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 996,
    height: 502,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAaklEQVR42j2MwQ6DIBQE+f9PrIea2GB5SoMg+hCbKW0TD3vZ3RkzDA9uXce971FVnBOstRy1cp5vTAihlQ7vPZseyOwREWKMpJQwpRRyzj86LJFxfF7ANM2Y2lR7G7+HJe+81n/8qkjc+AA48HnYpif8tQAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M2/vectorization-bts-2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/vectorization-bts-2.2b20c58b.png");}),
"[project]/public/images/M2/vectorization-bts-2.png.mjs { IMAGE => \"[project]/public/images/M2/vectorization-bts-2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$vectorization$2d$bts$2d$2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/vectorization-bts-2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$vectorization$2d$bts$2d$2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1004,
    height: 500,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAa0lEQVR42h3LCwqDIACAYe9/wdaKSaMwnzjSmTr4Fx3gEzlnoo+oVbEsC7tSGK1J9UdtHdF7J8YP4zgipeQ1Tch5JqeD8/wiaq0YY9CXaq1RSrmBsTtOvxHrtjEMD56XNNZypIwPAe8dwWn+CH14MAs5WxkAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/vector-dot-product.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/vector-dot-product.3a3d6462.png");}),
"[project]/public/images/M2/vector-dot-product.png.mjs { IMAGE => \"[project]/public/images/M2/vector-dot-product.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$vector$2d$dot$2d$product$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/vector-dot-product.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$vector$2d$dot$2d$product$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1020,
    height: 516,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAQUlEQVR42m2MSQ4AIAgD/f9vPbhDq6IkHiRp0jADgXdUwdLFooCvGQziwCG6Aj78CLKuY+pMdVjf8S8m7J6bfIUJroB+sCajUJ8AAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/gradient-descent-vector-notation.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-vector-notation.5b62a966.png");}),
"[project]/public/images/M2/gradient-descent-vector-notation.png.mjs { IMAGE => \"[project]/public/images/M2/gradient-descent-vector-notation.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$vector$2d$notation$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/gradient-descent-vector-notation.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$vector$2d$notation$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 999,
    height: 559,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAdElEQVR42h2LywrCMBQF8/+/JCIFceNShIoIIkKF2EducvNolDG4OgNnxgzDCwnKuAi+bakfnqPweAtzrBhrLbMTQszkdf0LS8hM7Uz1i1FV7DjhWh1SIZeK8401URqb471wuCW63rM9e7pLYtcrm5Owv0Z+iK544dEny2gAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/gradient-descent-for-multiple-regression.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-for-multiple-regression.790a6167.png");}),
"[project]/public/images/M2/gradient-descent-for-multiple-regression.png.mjs { IMAGE => \"[project]/public/images/M2/gradient-descent-for-multiple-regression.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$for$2d$multiple$2d$regression$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/gradient-descent-for-multiple-regression.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$for$2d$multiple$2d$regression$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 996,
    height: 562,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAkUlEQVR42g2MywqCQAAA/f8vCapDJJ071KkHSaeIgnwk62Iqu+4j02lPAwMzkagEtcwQ7xtFdidNX0jxRHcPvCmIjLV8yi26nuPaHUor6nJPK2b4JiaapglvDc71/AaH8R4pK5qmDl6Fg3NUlQyl5jsMgT15ntO2DeM4EsWJZnXuiC+KdWJYnjSLQ8fyqNhcLX8a45Qt7w0rNwAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M2/lab-notation.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/lab-notation.6df8eb24.png");}),
"[project]/public/images/M2/lab-notation.png.mjs { IMAGE => \"[project]/public/images/M2/lab-notation.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$lab$2d$notation$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/lab-notation.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$lab$2d$notation$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1442,
    height: 690,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAATUlEQVR42k2NCwrAMAhDe/+zFlqK9Qczq0LHhBDQZ9KICGstMDNEBGZWcndEBBrOqGopl+lzzoI/4P+190bvHXT8uUAebkVWjjEqKRNeNC184iYlIbMAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/matrix-X.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/matrix-X.0cfd7a56.png");}),
"[project]/public/images/M2/matrix-X.png.mjs { IMAGE => \"[project]/public/images/M2/matrix-X.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$matrix$2d$X$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/matrix-X.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$matrix$2d$X$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 954,
    height: 537,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAZUlEQVR42j2NSwoAIQxDvf8xVUTFhSL+0QzpYgoh0KSvqtYKrbXIGCNeSsGcE+ccKBastfDeI8aIEIJ4SgljDKj3Hu690m6tYe+N3ju45ygGay0RAzrxf4EXRDrnBJ9zBt+SxuIHL1GZmmBxts8AAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/parameter-w-b.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/parameter-w-b.29863469.png");}),
"[project]/public/images/M2/parameter-w-b.png.mjs { IMAGE => \"[project]/public/images/M2/parameter-w-b.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$parameter$2d$w$2d$b$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/parameter-w-b.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$parameter$2d$w$2d$b$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 952,
    height: 534,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAU0lEQVR42lWMSwrAIAxEvf89XSaKaCJRMm0sLjrwYGA+SVWRcz6UUmBm2HvD3RFKYWqtICK01tB7x1rrX4jFJR5veApzThATmBljDIjIhwrsfXoACWR8lM9pGLkAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/cost-function-formulae.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/cost-function-formulae.a588d0d2.png");}),
"[project]/public/images/M2/cost-function-formulae.png.mjs { IMAGE => \"[project]/public/images/M2/cost-function-formulae.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$cost$2d$function$2d$formulae$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/cost-function-formulae.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$cost$2d$function$2d$formulae$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 895,
    height: 254,
    blurWidth: 8,
    blurHeight: 2,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAACCAYAAABllJ3tAAAAMklEQVR42k2JOQoAQAjE/P9fRbDwQIZZtNpUCZGqYnczIqiqNLPrBQDF3W9mJtf/nhk+AFU+Gm2CHH4AAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/gradient-descent-formulae.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-formulae.17a9ce50.png");}),
"[project]/public/images/M2/gradient-descent-formulae.png.mjs { IMAGE => \"[project]/public/images/M2/gradient-descent-formulae.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$formulae$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/gradient-descent-formulae.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$formulae$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1027,
    height: 680,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAZElEQVR42j2NWQoAIQxDvf85VRBBXBDXMWMKM4V8pH1J1XMOQggwxkBrDWstnHPw3mOtBXUA7L0xxkDOGTFGzDlld25YXf1ArRWlFPTexROUBpKEUkryrrUmnhKAJOt5/ACGOC/Q75tKt4uIQAAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M2/gradient-descent-for-multiple-variables.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-for-multiple-variables.47ca840d.png");}),
"[project]/public/images/M2/gradient-descent-for-multiple-variables.png.mjs { IMAGE => \"[project]/public/images/M2/gradient-descent-for-multiple-variables.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$for$2d$multiple$2d$variables$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/gradient-descent-for-multiple-variables.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$for$2d$multiple$2d$variables$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1000,
    height: 316,
    blurWidth: 8,
    blurHeight: 3,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAADCAYAAACuyE5IAAAATElEQVR42h2MKQ4AIRDA+P8nIQgOhQHCOd1lmlTU1Nx7iTFirVWdc3jvKaWw98aIiEYIgZwztVZ678w5Oedg+BljkFJSW2ustXjnxwePzlu6IsoGPgAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M2/gradient-descent-for-multiple-variables-plot.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-for-multiple-variables-plot.7dcd7e2b.png");}),
"[project]/public/images/M2/gradient-descent-for-multiple-variables-plot.png.mjs { IMAGE => \"[project]/public/images/M2/gradient-descent-for-multiple-variables-plot.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$for$2d$multiple$2d$variables$2d$plot$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/gradient-descent-for-multiple-variables-plot.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$gradient$2d$descent$2d$for$2d$multiple$2d$variables$2d$plot$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1408,
    height: 447,
    blurWidth: 8,
    blurHeight: 3,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAADCAYAAACuyE5IAAAARklEQVR42h2KSQ6AQAyA5v+PNcZYpztWb0BY2xyLxMyoKsQC2YZH0N2sSzaWPQKZRcysyTT+ttST83GOWxENImfQj324eAE67V44H9ZgoQAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M2/quiz-q1.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/quiz-q1.6a97981d.png");}),
"[project]/public/images/M2/quiz-q1.png.mjs { IMAGE => \"[project]/public/images/M2/quiz-q1.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$quiz$2d$q1$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/quiz-q1.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$quiz$2d$q1$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1092,
    height: 692,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAhklEQVR42mXNOw6CQAAA0T2GuogkegIVa+MH0HMoHINKKqnxZnwWbyBhk6UfFewsXjfJiJW7ITidOXg+Rz/oeR/LtYuc2og4jimLgjzPUUpRVSX1s+aepoylhbglCU3zwhhD1xl026K1JsseQ7Dd7bmEEdfoJxx8V6OJREjbwZkv/lj2rA/eDcpaZy1vgOcAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M2/quiz-q2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/quiz-q2.03be2f6e.png");}),
"[project]/public/images/M2/quiz-q2.png.mjs { IMAGE => \"[project]/public/images/M2/quiz-q2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$quiz$2d$q2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/quiz-q2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$quiz$2d$q2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1072,
    height: 367,
    blurWidth: 8,
    blurHeight: 3,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAADCAYAAACuyE5IAAAARklEQVR42j2KyQ3AIAwEKSMB7PjAidJ/gYvsB4/RrjTT6BGIGlgU+ZNBjKuP2qYeiO+HmB95j1lB0qY4LF7oWmDVCvqkE20s4yByrazCFQAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M2/quiz-q3.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/quiz-q3.c11960cb.png");}),
"[project]/public/images/M2/quiz-q3.png.mjs { IMAGE => \"[project]/public/images/M2/quiz-q3.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$quiz$2d$q3$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M2/quiz-q3.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M2$2f$quiz$2d$q3$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1075,
    height: 346,
    blurWidth: 8,
    blurHeight: 3,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAADCAYAAACuyE5IAAAAQklEQVR42j3LyxHAIAgEUMpIJCIgof8ON5EZPbzLfsgjEZkY5uhDD+5SSCxg8cJmQH1C1Er7y6sxaK12uOznzU8NPj2aIJ1D2QkDAAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/linear-regression.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/linear-regression.bff57883.png");}),
"[project]/public/images/M1/linear-regression.png.mjs { IMAGE => \"[project]/public/images/M1/linear-regression.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$linear$2d$regression$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/linear-regression.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$linear$2d$regression$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 807,
    height: 442,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAIAAAA8r+mnAAAAW0lEQVR42hXMQQ6AIAwAQf7/QS8eNDYxtqUFrREwgHjezbjWmtl9sOzkhVkQ05N7744kHuADRdMrABjxW8ofYq5y3iI6dkWIQc2s1upW1AXDYGhdeJ48bCml4X+rvFokSFQuTAAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/linear-regression-2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/linear-regression-2.b65a79c7.png");}),
"[project]/public/images/M1/linear-regression-2.png.mjs { IMAGE => \"[project]/public/images/M1/linear-regression-2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$linear$2d$regression$2d$2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/linear-regression-2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$linear$2d$regression$2d$2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 976,
    height: 486,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAZklEQVR42k3KOw7CMBBAQd//hEEIUCiCEBB/17E3Gz9Ex9Tjcs5M08TtcqaFFyLC8ok8fWK3A/cLp+udeV7Q2pFSiLHgmyE6cLolHmFjXTsaGilGkvdUNWyAG7uQakOj0t+Jw4x/X15IevoCBJqkAAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/cost-function.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/cost-function.e5010460.png");}),
"[project]/public/images/M1/cost-function.png.mjs { IMAGE => \"[project]/public/images/M1/cost-function.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$cost$2d$function$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/cost-function.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$cost$2d$function$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1270,
    height: 638,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAIAAAA8r+mnAAAAUUlEQVR42h2MWw6AIAzAuP81Tfwxjr3HmCSi/W3TpkxPVc5yE0QkJP5p13EKUET4GNip39gBzKwJsamEe2QiMovuqKqajzT3zAnwXUy38LXWC9UvW4cGf/1hAAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/goal.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/goal.1b7d92c7.png");}),
"[project]/public/images/M1/goal.png.mjs { IMAGE => \"[project]/public/images/M1/goal.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$goal$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/goal.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$goal$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 526,
    height: 461,
    blurWidth: 8,
    blurHeight: 7,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAHCAYAAAA1WQxeAAAAj0lEQVR42m1OSQ7CMAzM//8IqKCGZiEbzdI2ZIhzgUNHsix5NrMQApZlwb7vOANLKeMxc0j9gnEepRTUWn8CrTUu1xum6Q4f3ojdIO2KUBrqpwvWNWJ+ChhjQHUxJXChYK1Faw0s5wylFKSUPWWCcw4xxlFFHCOXEGIcaXvvsW0bjuMYw4ggJ+d8xP4/SPgCR67XZIErfScAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/simplified-linear-regression.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/simplified-linear-regression.810c5ba0.png");}),
"[project]/public/images/M1/simplified-linear-regression.png.mjs { IMAGE => \"[project]/public/images/M1/simplified-linear-regression.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$simplified$2d$linear$2d$regression$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/simplified-linear-regression.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$simplified$2d$linear$2d$regression$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 412,
    height: 438,
    blurWidth: 8,
    blurHeight: 8,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAs0lEQVR42jWP3XKDIBCFff/369SrtNEm0QRFQYOywFfEyZk9F7uzc36qlBIhBNq2pa7rQmMM+VxQ7fvO46lQamAcXqyb8LSC3eP5cCgYF4pKdAZrLevbITGRh8p7j4jwwTzP9HpBLYKTeCocPKycc/xeO76+r3Rdz7ZtVMV/1DRNw+XyQ3PveA0auyxM05QtJPDOnscyap3zCGoNzDlXzMqlRfN3o709uPcq5wmsPuHD2fMf+o72F/+HxvUAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/fwx-vs-jw.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/fwx-vs-jw.e0086e08.png");}),
"[project]/public/images/M1/fwx-vs-jw.png.mjs { IMAGE => \"[project]/public/images/M1/fwx-vs-jw.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$fwx$2d$vs$2d$jw$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/fwx-vs-jw.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$fwx$2d$vs$2d$jw$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1059,
    height: 630,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAkUlEQVR42h3JSQ6CMABA0d7/Aibqwug1UGP0CE5gAgGHMIhS2lIq6Ne4eKsnoihitz+w/QnDiDhOCAKfo38iTs4Iay3WtihlkXWDcw5tDNb19O8PQimDrAxN43ia9p9KKV5dh2x7RCk1aXbnUUmqWlNWNZc0pyhKlNaIySpj6F2ZbgrG8xsDL2W0zBkvMmbrgi8eE5MhEiPl6AAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/fwx-vs-jw-2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/fwx-vs-jw-2.1d75c3f4.png");}),
"[project]/public/images/M1/fwx-vs-jw-2.png.mjs { IMAGE => \"[project]/public/images/M1/fwx-vs-jw-2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$fwx$2d$vs$2d$jw$2d$2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/fwx-vs-jw-2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$fwx$2d$vs$2d$jw$2d$2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1069,
    height: 602,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAhElEQVR42h3MWwrCMBRF0cx/DqIgQh2IqNVxVCzGNsnNo0nKNvbjwIYDS9VaMcYi4sk5k0vBWUsIgVIqKsZIWhZiyljfOqXtXNd1mxIn6K9l0J7ZhSYJtgmlSf8p1+jXx/CeHGHJTBIYRs2oZ2xrde4Nx7vQPT2n3rO/enYX4XDzdA/hB7nVlsV1M7tdAAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/fwx-vs-jw-3.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/fwx-vs-jw-3.68922395.png");}),
"[project]/public/images/M1/fwx-vs-jw-3.png.mjs { IMAGE => \"[project]/public/images/M1/fwx-vs-jw-3.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$fwx$2d$vs$2d$jw$2d$3$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/fwx-vs-jw-3.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$fwx$2d$vs$2d$jw$2d$3$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1069,
    height: 604,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAe0lEQVR42h2MWwoCMQxFu/89KIz64T4cBl2LD8yrTemUO2kDF8LNyUmtNYgIshn6vqP3DlOBe8GYZHGo7qi1onidezad4ASYOQqB5AwN2FQh0Q3zgJKEjiIckMa35oLf94M/URgdaXm8sWyE25NwWgnnlXHZGNfI/cU4AOs5lybpeptUAAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/jw-plot.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/jw-plot.a56ffc3a.png");}),
"[project]/public/images/M1/jw-plot.png.mjs { IMAGE => \"[project]/public/images/M1/jw-plot.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$jw$2d$plot$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/jw-plot.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$jw$2d$plot$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1072,
    height: 598,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAa0lEQVR42hWKWwoCMRDAev8LCSoieAVxxX9BFOlzZjplIdv9CPlIgrtTa0VEGGOQmpJTxKQyTAh77L1j3VHrqDSkRFZX1uH7UEglEaeLCqU1fp83+f/FzAjne+a4KKfJ4WHTxmURrk/l9lI2tI544BSmIxsAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/minimizing-jw.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/minimizing-jw.765a8802.png");}),
"[project]/public/images/M1/minimizing-jw.png.mjs { IMAGE => \"[project]/public/images/M1/minimizing-jw.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$minimizing$2d$jw$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/minimizing-jw.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$minimizing$2d$jw$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1070,
    height: 601,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAcklEQVR42hWKOwpCMRQFs//tKFhaiFqLCILaKBEFSTTJveYDmZdXDIczjIkp8QthEBH946LgnUNiIGvCeO+x1hJGVGtFRUhfRxvbSsbMUubTGr13QhKe9wef1xtVxWyvmd2tsB9sLoXVSVkcEsujsD4LE5BYeLEN9/tOAAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/cost-function-minimization-goals.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/cost-function-minimization-goals.5941ca87.png");}),
"[project]/public/images/M1/cost-function-minimization-goals.png.mjs { IMAGE => \"[project]/public/images/M1/cost-function-minimization-goals.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$cost$2d$function$2d$minimization$2d$goals$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/cost-function-minimization-goals.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$cost$2d$function$2d$minimization$2d$goals$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1069,
    height: 601,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAZUlEQVR42i2MWQ6EIBQEvf+5Js5MvIDrFUAUfGymBOJHpzudSnXHcbLvhhAidTt3kXIm3ze+fJ2IoJSido21Fu89KSVifAFjqiG01lq3nYulAf/14rc4hk34jGeJo5+F7yL0k+UB3Bh54dHnhfQAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/3d-surface-plot-of-jwb.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/3d-surface-plot-of-jwb.c88c3f29.png");}),
"[project]/public/images/M1/3d-surface-plot-of-jwb.png.mjs { IMAGE => \"[project]/public/images/M1/3d-surface-plot-of-jwb.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$3d$2d$surface$2d$plot$2d$of$2d$jwb$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/3d-surface-plot-of-jwb.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$3d$2d$surface$2d$plot$2d$of$2d$jwb$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1068,
    height: 600,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAeUlEQVR42iWM0QqCMAAA/f+/6QtMCkMDC5IotQgqCCtzrLU52k7Fh3s5jgucc9iuGzBorRD3Ei0aRj8SWGtpnjWPa8a2iFhkM4pLjGxfuP8QeO8xxvD+1Gx2a8I0pbqdUUpOh+SkSSpNfPwyz1vCXBDtJcuDYlX+6AENknakyz2oagAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/mt-fuji-contour-map.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/mt-fuji-contour-map.1f6df426.png");}),
"[project]/public/images/M1/mt-fuji-contour-map.png.mjs { IMAGE => \"[project]/public/images/M1/mt-fuji-contour-map.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$mt$2d$fuji$2d$contour$2d$map$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/mt-fuji-contour-map.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$mt$2d$fuji$2d$contour$2d$map$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1068,
    height: 600,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAh0lEQVR42gWA3QrBUACAz2OJF1jtQmc6WIsypq22ZmlFbaXkf4SLsSJx40LJnvCTeL7vdB1J0DOIpKRTraGqFYZmm+/vgzgXC7yBxtxqMVUGWeiTWia2rpHne0R+W+P2dSaqQdKUnEKPmWMTqDrXYoPYvUrcZEkUH4ijjMBf4Y+PjNIL20fJH8SbSw9eHQE8AAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/cost-function-3d-plot.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/cost-function-3d-plot.ca2a2c23.png");}),
"[project]/public/images/M1/cost-function-3d-plot.png.mjs { IMAGE => \"[project]/public/images/M1/cost-function-3d-plot.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$cost$2d$function$2d$3d$2d$plot$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/cost-function-3d-plot.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$cost$2d$function$2d$3d$2d$plot$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1066,
    height: 597,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAdElEQVR42kXKsQrCMBhF4b7/uzg4OgqKVHAWoRUqilBpkkqSP41Nji0depcLh6/IORNCYHCWOH2rOlqt8SLMK2bgxWO/BpliZxTKWoYYV5BSwjrH8/2ivjdo0/MbxwWUlXCuhcO1Z3t6sNk37C4fjjdPWQX+z7l5DKpBFjgAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/contour-plot-of-jwb.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/contour-plot-of-jwb.63c71255.png");}),
"[project]/public/images/M1/contour-plot-of-jwb.png.mjs { IMAGE => \"[project]/public/images/M1/contour-plot-of-jwb.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$contour$2d$plot$2d$of$2d$jwb$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/contour-plot-of-jwb.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$contour$2d$plot$2d$of$2d$jwb$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1063,
    height: 599,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAlUlEQVR42iWMMQuCQABG/eFtbQ31Ey6KGmuQQigKp1qCIFrcokhR1FPvFE/xpfTBN73Hs+hX1zWmLDFVRZTEhElCmheYpsXqug6lFFqmmF4MZYwvM3b2lvlsijUUmrZBa0VRFKRSkmuNe3LYLMRfGCp5D5+ex+X64P0JWIslk9EYa3/XDLdvGSs3QBx9xDlCHL7MnRc/8rGP4+cewqUAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/cost-function-visualization.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/cost-function-visualization.ba4d5082.png");}),
"[project]/public/images/M1/cost-function-visualization.png.mjs { IMAGE => \"[project]/public/images/M1/cost-function-visualization.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$cost$2d$function$2d$visualization$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/cost-function-visualization.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$cost$2d$function$2d$visualization$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1069,
    height: 599,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAcklEQVR42iXKsQoCMRCE4bz/c1ja2Z+i2NkKh4KK2BjQ3OV2w2qIvws3zcB8E2qtmBmShTQV3jkT08CghW/7EebDhzQKoxREvR3FN/f50FpDVXnGyOV6J+fJ0dUTNr2yOxvdMbPcP1h0N1aHF+ve2J6MP7uMeLZRwPmcAAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/note.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/note.482df74c.png");}),
"[project]/public/images/M1/note.png.mjs { IMAGE => \"[project]/public/images/M1/note.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$note$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/note.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$note$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 959,
    height: 662,
    blurWidth: 8,
    blurHeight: 6,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAGCAYAAAD+Bd/7AAAAfUlEQVR42k2OXRKDIAyEOUZH5R+kCGg7be9/tK0bX3zILCS7X6LGfuDz/aFuDc4HuBBhT2UZ56Fqazheb1BzeSLmFSFlhJjErLR1WIyVBIdsztrgMc2YFn0ZiCI+rUWKf5oYVES1sWPr/dQh79o6Sq1CVMRwN5PE+9uRnP0BLplD9A0rDJQAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/note-2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/note-2.f2b0977f.png");}),
"[project]/public/images/M1/note-2.png.mjs { IMAGE => \"[project]/public/images/M1/note-2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$note$2d$2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/note-2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$note$2d$2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1503,
    height: 187,
    blurWidth: 8,
    blurHeight: 1,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAABCAYAAADjAO9DAAAAH0lEQVR42mPQ1df/D8Lqmlr/VdTU/yurqgFpNTANwgC5+Qu0KSLpMwAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/gradient-descent-1.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-1.0f5274d7.png");}),
"[project]/public/images/M1/gradient-descent-1.png.mjs { IMAGE => \"[project]/public/images/M1/gradient-descent-1.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$1$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/gradient-descent-1.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$1$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1071,
    height: 599,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAj0lEQVR42gGEAHv/APz9/f/4/P7/9vr8//Hq6f/8+Pb/7ujf/4mOX/9WaD//APb4+f/2+e//5enC/8rFkv/N25r/ztGg/77Ww//K0sf/AOLw+P/r9PH/y+rh/6LKy/+54tn/xOTj/+n39//9/f3/ALnU9P+81vX/vdXz/7PI6v+yyev/t9Dy/7rU9f+81fT/AJpwSW0oUW0AAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/gradient-descent-algorithm.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-algorithm.d73bf8b6.png");}),
"[project]/public/images/M1/gradient-descent-algorithm.png.mjs { IMAGE => \"[project]/public/images/M1/gradient-descent-algorithm.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$algorithm$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/gradient-descent-algorithm.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$algorithm$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1053,
    height: 522,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAcElEQVR42g3NSwqDMBRA0ex/dx1IpUjRSqF+EvPyUxPDrfMDR726jrZpaR5PxvcHvWiWeCIu4W1CiWjmcWDqe865J8uE2zMhJI50oEIKiP6yzQM5rjgf2axHRCiloMwtl3UleMt1ZbTb+d2NMUKtlT/rP3kw02wnEwAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/gradient-descent-algorithm-2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-algorithm-2.0e11f497.png");}),
"[project]/public/images/M1/gradient-descent-algorithm-2.png.mjs { IMAGE => \"[project]/public/images/M1/gradient-descent-algorithm-2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$algorithm$2d$2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/gradient-descent-algorithm-2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$algorithm$2d$2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 681,
    height: 293,
    blurWidth: 8,
    blurHeight: 3,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAADCAYAAACuyE5IAAAASklEQVR42j2KMQpAIQxDvf8RdVEcdOlQKB1aSj4tfANJ4PHaOQe9d8w5Mcao33sjuaqiuTuIqLrWwr0XZoY/LUdEwMwlpBgRT/gAVedbBqsa7EUAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/learning-rate-1.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/learning-rate-1.e9c8c0d7.png");}),
"[project]/public/images/M1/learning-rate-1.png.mjs { IMAGE => \"[project]/public/images/M1/learning-rate-1.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$learning$2d$rate$2d$1$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/learning-rate-1.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$learning$2d$rate$2d$1$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1053,
    height: 518,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAaklEQVR42h2MWw7CIAAEuf8ZlcQYbNKCCJRHy2Ok3a/NZmdEa424B0opXBlj4L2nnJV8dETOmW1dcT/LMU+997trY3AhIWKMKKXQmyanRGsVZ6cxnORYEWmO8iGRT8nyWbiAr3W8XxZjdv7md3q/7EzZhQAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/local-minimum.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/local-minimum.fce6e2a0.png");}),
"[project]/public/images/M1/local-minimum.png.mjs { IMAGE => \"[project]/public/images/M1/local-minimum.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$local$2d$minimum$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/local-minimum.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$local$2d$minimum$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1008,
    height: 513,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAW0lEQVR42lWM2w6DMAxD+/9/yja0AqNJSpqzAE9Y8k2yXNwHIsoI6B6kPVBUlbr+kO60Zlgy9AD3e+AZljrz/Uxs8xvZVtx2RpIISqSYNOr0Qpads4t1/BjXwx9gb31MtnUZ8AAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/learning-rate-2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/learning-rate-2.a335bc01.png");}),
"[project]/public/images/M1/learning-rate-2.png.mjs { IMAGE => \"[project]/public/images/M1/learning-rate-2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$learning$2d$rate$2d$2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/learning-rate-2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$learning$2d$rate$2d$2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1008,
    height: 513,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAa0lEQVR42h3JyQ6CMBgA4b7/65kYezBAXH+6QqpdBIfAHObyqZQSwVmCGfGvO945rLX4USiloPYFF0lhZs2ZX6sH7NX2Rz3FoK893XAjxnjg55tZl4U6zaiuHzidL2itERFaaxg/8X4IkzFsbXd5aTQtQ4IAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/formula-recap.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/formula-recap.bf67ece1.png");}),
"[project]/public/images/M1/formula-recap.png.mjs { IMAGE => \"[project]/public/images/M1/formula-recap.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$formula$2d$recap$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/formula-recap.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$formula$2d$recap$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 947,
    height: 484,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAZUlEQVR42j2MSQ7DIAAD+f8jeyAhLIVQgsQm0FT00Jtle0bknLHWYowhhEBKCSnlL/feEbtQSnGpC601GyilMOdkrYUYY+C955Qnx+vgiQ8xRmqt7E3sV2sN7wLe3sT7g3Pvv+ULSsd552nyMWIAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/gradient-descent-derivation.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-derivation.97d82a01.png");}),
"[project]/public/images/M1/gradient-descent-derivation.png.mjs { IMAGE => \"[project]/public/images/M1/gradient-descent-derivation.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$derivation$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/gradient-descent-derivation.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$derivation$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1047,
    height: 529,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAZ0lEQVR42iXMWw7CMBBD0e5/lwiECmleQyYPkuoyqJ+27LM97jd2H3nXRewnoU2O0vA6iHWwzTktfPFtIeMka8e5HfkkVDNbESH9SxOyDZINDykkU6R1E5a96sTpJYRUcK8nkgNNhR+yQnsd/LZfVgAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/gradient-descent-formulae.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-formulae.cae3834e.png");}),
"[project]/public/images/M1/gradient-descent-formulae.png.mjs { IMAGE => \"[project]/public/images/M1/gradient-descent-formulae.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$formulae$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/gradient-descent-formulae.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$formulae$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1049,
    height: 522,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAY0lEQVR42jWM0Q6CMAwA9/9fKQ/aAlmYshWprCeYeMm9XS7lnBERVAVRRaaZo3f+JHenlAW9D9TnQmsv/OOYGRFB6mfdauUx3CjzxG4rWzNGHfHdSVd1We3NuvlvHx700ziCL6bGe5uwxXIgAAAAAElFTkSuQmCC"
};
}),
"[project]/public/images/M1/gradient-descent-on-sq-error-cost-function.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-on-sq-error-cost-function.12187a33.png");}),
"[project]/public/images/M1/gradient-descent-on-sq-error-cost-function.png.mjs { IMAGE => \"[project]/public/images/M1/gradient-descent-on-sq-error-cost-function.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$on$2d$sq$2d$error$2d$cost$2d$function$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/gradient-descent-on-sq-error-cost-function.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$on$2d$sq$2d$error$2d$cost$2d$function$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1049,
    height: 522,
    blurWidth: 8,
    blurHeight: 4,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAECAYAAACzzX7wAAAAd0lEQVR42hXMQQrCMBRF0ex/Fe5BBHEguAAn2s5EQavVVtGY5hea5t/+Dt7o8o5TVf7e420SI93zSi9CKyONZFxKiW/b8KkOnG57ymJJ/TgSQmDIipuFaM/6dWG1W7PYbCnPhWmBuTkFcs5EY+/vH5Wti8JgsnUmG2h4tx260S0AAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/batch-gradient-descent.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/batch-gradient-descent.bc565fe1.png");}),
"[project]/public/images/M1/batch-gradient-descent.png.mjs { IMAGE => \"[project]/public/images/M1/batch-gradient-descent.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$batch$2d$gradient$2d$descent$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/batch-gradient-descent.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$batch$2d$gradient$2d$descent$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 834,
    height: 513,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAeUlEQVR42j2MWw6CMBQF2f/y+FCiEoMPoiiUWwRraTu2kHB+z8xkxFlrGQbNz3zppWdfFLRdSwiBzBjDNI3IvUJeD0QU1aWibpoVSIV5tsi7QZRaSuP4YYriBjjn0FrzjNbheKZrFd77dK1AIhNwKkvyfMftWi9S2h+AhZl1BtJNegAAAABJRU5ErkJggg=="
};
}),
"[project]/public/images/M1/quiz-1.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/quiz-1.b31cc12b.png");}),
"[project]/public/images/M1/quiz-1.png.mjs { IMAGE => \"[project]/public/images/M1/quiz-1.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$quiz$2d$1$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/quiz-1.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$quiz$2d$1$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1087,
    height: 682,
    blurWidth: 8,
    blurHeight: 5,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAIAAAD38zoCAAAAZUlEQVR42j2LSw5AMAAFewz6L8o9EEQdRRucACucstveQpOqZFZv3oC6abUxszajmghjmFKISYow2I/TOWetve4ngegHtF2/rJsvhlGF7ycIz0RRMpFRLjyI0CgYl1WVSxlWGMUL+4UafVi+00AAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/quiz-2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/quiz-2.5494525b.png");}),
"[project]/public/images/M1/quiz-2.png.mjs { IMAGE => \"[project]/public/images/M1/quiz-2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$quiz$2d$2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/quiz-2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$quiz$2d$2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 940,
    height: 367,
    blurWidth: 8,
    blurHeight: 3,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAADCAYAAACuyE5IAAAASklEQVR42jXMQQ6AIAxEUY6hUEpLEeP9D/hFEhcvmc2fVK3TYi4XzQNR5SyyHbmQpBnjfvAxsRhYj029k6WSStVVOmrOt/96PywvNgQgicGYUvMAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/gradient-descent-note-1.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-note-1.18a18c28.png");}),
"[project]/public/images/M1/gradient-descent-note-1.png.mjs { IMAGE => \"[project]/public/images/M1/gradient-descent-note-1.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$note$2d$1$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/gradient-descent-note-1.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$note$2d$1$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1536,
    height: 540,
    blurWidth: 8,
    blurHeight: 3,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAADCAYAAACuyE5IAAAARElEQVR42j3ISQoAIRDAQP//U72IiGsvGRzQQC4V5pzEGEkpkXOmlELvnbUWIkJQVcYY71ore2+OmxnB3X+42Frj2O0D99hdmvKJeNwAAAAASUVORK5CYII="
};
}),
"[project]/public/images/M1/gradient-descent-note-2.png (static in ecmascript)", ((__turbopack_context__) => {

__turbopack_context__.v("/_next/static/media/gradient-descent-note-2.c3a09cee.png");}),
"[project]/public/images/M1/gradient-descent-note-2.png.mjs { IMAGE => \"[project]/public/images/M1/gradient-descent-note-2.png (static in ecmascript)\" } [app-rsc] (structured image object with data url, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>__TURBOPACK__default__export__
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$note$2d$2$2e$png__$28$static__in__ecmascript$29$__ = __turbopack_context__.i("[project]/public/images/M1/gradient-descent-note-2.png (static in ecmascript)");
;
const __TURBOPACK__default__export__ = {
    src: __TURBOPACK__imported__module__$5b$project$5d2f$public$2f$images$2f$M1$2f$gradient$2d$descent$2d$note$2d$2$2e$png__$28$static__in__ecmascript$29$__["default"],
    width: 1536,
    height: 250,
    blurWidth: 8,
    blurHeight: 1,
    blurDataURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAABCAYAAADjAO9DAAAALElEQVR42gEhAN7/AP39/f/5+fn/+/v7//T09P/y8vL/+vr6//39/f/+/v7/BOcfXSiOgnMAAAAASUVORK5CYII="
};
}),
"[project]/.source/index.ts [app-rsc] (ecmascript)", ((__turbopack_context__) => {
"use strict";

// @ts-nocheck -- skip type checking
__turbopack_context__.s([
    "docs",
    ()=>docs
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$intro$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/content/ml/supervised-learning/intro/index.mdx.js?collection=docs&hash=1759422546180 [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$labs$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/content/ml/supervised-learning/labs/index.mdx.js?collection=docs&hash=1759422546180 [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$multiple$2d$linear$2d$regression$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/content/ml/supervised-learning/multiple-linear-regression/index.mdx.js?collection=docs&hash=1759422546180 [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$linear$2d$regression$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/content/ml/supervised-learning/linear-regression/index.mdx.js?collection=docs&hash=1759422546180 [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$gradient$2d$descent$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/content/ml/supervised-learning/gradient-descent/index.mdx.js?collection=docs&hash=1759422546180 [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/content/ml/supervised-learning/index.mdx.js?collection=docs&hash=1759422546180 [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/content/ml/index.mdx.js?collection=docs&hash=1759422546180 [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$fumadocs$2d$mdx$2f$dist$2f$runtime$2f$next$2f$index$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/fumadocs-mdx/dist/runtime/next/index.js [app-rsc] (ecmascript) <locals>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$fumadocs$2d$mdx$2f$dist$2f$chunk$2d$AUOOMFAI$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/fumadocs-mdx/dist/chunk-AUOOMFAI.js [app-rsc] (ecmascript)");
;
;
;
;
;
;
;
;
const docs = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$fumadocs$2d$mdx$2f$dist$2f$chunk$2d$AUOOMFAI$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["_runtime"].docs([
    {
        info: {
            "path": "index.mdx",
            "fullPath": "content/ml/index.mdx"
        },
        data: __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__
    },
    {
        info: {
            "path": "supervised-learning/index.mdx",
            "fullPath": "content/ml/supervised-learning/index.mdx"
        },
        data: __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__
    },
    {
        info: {
            "path": "supervised-learning/gradient-descent/index.mdx",
            "fullPath": "content/ml/supervised-learning/gradient-descent/index.mdx"
        },
        data: __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$gradient$2d$descent$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__
    },
    {
        info: {
            "path": "supervised-learning/linear-regression/index.mdx",
            "fullPath": "content/ml/supervised-learning/linear-regression/index.mdx"
        },
        data: __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$linear$2d$regression$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__
    },
    {
        info: {
            "path": "supervised-learning/multiple-linear-regression/index.mdx",
            "fullPath": "content/ml/supervised-learning/multiple-linear-regression/index.mdx"
        },
        data: __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$multiple$2d$linear$2d$regression$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__
    },
    {
        info: {
            "path": "supervised-learning/labs/index.mdx",
            "fullPath": "content/ml/supervised-learning/labs/index.mdx"
        },
        data: __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$labs$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__
    },
    {
        info: {
            "path": "supervised-learning/intro/index.mdx",
            "fullPath": "content/ml/supervised-learning/intro/index.mdx"
        },
        data: __TURBOPACK__imported__module__$5b$project$5d2f$content$2f$ml$2f$supervised$2d$learning$2f$intro$2f$index$2e$mdx$2e$js$3f$collection$3d$docs$26$hash$3d$1759422546180__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__
    }
], [
    {
        "info": {
            "path": "supervised-learning/meta.json",
            "fullPath": "content/ml/supervised-learning/meta.json"
        },
        "data": {
            "pages": [
                "index",
                "linear-regression",
                "gradient-descent",
                "multiple-linear-regression",
                "labs"
            ]
        }
    }
]);
}),
"[project]/app/ml/layout.tsx [app-rsc] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>Layout
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$rsc$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/server/route-modules/app-page/vendored/rsc/react-jsx-dev-runtime.js [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$fumadocs$2d$ui$2f$dist$2f$layouts$2f$docs$2f$index$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i("[project]/node_modules/fumadocs-ui/dist/layouts/docs/index.js [app-rsc] (ecmascript) <locals>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$layout$2e$shared$2e$tsx__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/layout.shared.tsx [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$source$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/source.ts [app-rsc] (ecmascript)");
;
;
;
;
function Layout({ children }) {
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$rsc$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$fumadocs$2d$ui$2f$dist$2f$layouts$2f$docs$2f$index$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__$3c$locals$3e$__["DocsLayout"], {
        tree: __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$source$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["source"].pageTree,
        ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$layout$2e$shared$2e$tsx__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["baseOptions"])(),
        children: children
    }, void 0, false, {
        fileName: "[project]/app/ml/layout.tsx",
        lineNumber: 7,
        columnNumber: 5
    }, this);
}
}),
];

//# sourceMappingURL=_7f1d6d33._.js.map