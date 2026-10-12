import { resources, readResource } from '../../lib/repository';
import type { APIRoute } from 'astro';
export function getStaticPaths() {
  return resources.map(resource => ({ params: { slug: resource.slug }, props: { resource } }));
}
export const GET: APIRoute = ({ props }) => new Response(readResource(props.resource.path), {
  headers: { 'Content-Type': 'text/plain; charset=utf-8' },
});
