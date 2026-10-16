const {chromium}=require('/Users/shaoshitong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');const path=require('path');const crypto=require('crypto');
(async()=>{
 const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1500,height:1000}});let errors=[];page.on('pageerror',e=>errors.push(String(e)));
 const response=await page.goto('http://127.0.0.1:18766/ranking.html');
 if(response.status()!==200)throw Error('Existing viewer is not served');
 const served=await response.text();const expected=fs.readFileSync(path.join(__dirname,'ranking.html'),'utf8');
 if(served!==expected)throw Error('Existing URL does not serve the refreshed page');
 await page.locator('#figure svg').waitFor();let results=[];
 for(const label of ['训练EM曲线','固定512题柱状图','51,713题终评表']){
  await page.getByRole('button',{name:label,exact:true}).click();
  const paths=await page.locator('#figure svg path').count();if(paths<10)throw Error('Incomplete chart '+label);
  const width=await page.locator('#figure svg').evaluate(el=>el.getBoundingClientRect().width);if(width<700)throw Error('Unexpected SVG width');
  results.push({label,paths,width});
 }
 await page.locator('input[type=range]').evaluate(el=>{el.value='150';el.dispatchEvent(new Event('input',{bubbles:true}))});
 if(await page.locator('#z').textContent()!=='150%')throw Error('Zoom control failed');
 await page.getByRole('button',{name:'适合窗口',exact:true}).click();
 await page.screenshot({path:path.join(__dirname,'viewer-final-table-qa.png'),fullPage:true});
 await browser.close();if(errors.length)throw Error(errors.join('\n'));
 const audit={url:'http://127.0.0.1:18766/ranking.html',served_page_sha256:crypto.createHash('sha256').update(served).digest('hex'),tabs:results,zoom_test:true,errors,read_only_browser_check:true};
 fs.writeFileSync(path.join(__dirname,'browser-verification.json'),JSON.stringify(audit,null,2));process.stdout.write(JSON.stringify(audit));
})().catch(e=>{process.stderr.write(String(e));process.exit(1)});
